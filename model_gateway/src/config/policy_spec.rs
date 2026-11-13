//! The routing flags compiled into a policy specification at start (the
//! design's step 2, release N). One converter over `RouterConfig`: the
//! command line, the launcher, the binding's keywords and the chart all
//! build one, so the four dialects share it by construction. The converter
//! writes each policy's expansion and a parameter only where the flag's
//! value differs from the stage's default, and its report names every flag
//! it consumed with the field it landed in: the start logs one deprecation
//! line per flag and counts it in `smg_config_deprecated_flags`. The flags
//! still decide in this release; the document says what they would read as.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    sync::LazyLock,
};

use serde_json::Value;
use smg_policy_controller::{
    spec::{
        Assignment, BestWithTemperatureParams, BindingSpec, BoundedLoadParams, BucketParams,
        ConsistentHashParams, IndexKind, IndexSpec, KvPressureParams, LeastLoadParams,
        OverloadSpec, Pairing, PickSpec, PickStage, PlanName, PlansSpec, PrefixAffinityParams,
        ProfileSpec, Recovery, ResidentWorkParams, ScoreSpec, ScoreStage, SessionsSpec,
        SpillParams, ViewSpec, WarmupSliceParams, MANUAL_SESSION_HEADER,
    },
    Phase, PolicySpec,
};

use super::types::{
    CacheIndexKind, ManualAssignmentMode, PdPairingMode, PolicyConfig, PowerOfTwoLoadMetric,
    RouterConfig, RoutingKeyOverrideConfig, RoutingMode, DEFAULT_ENCODE_POLICY,
};

/// The routing flags a front end saw on its command line, by clap id (the
/// `RoutingArgs` field name: `cache_threshold`). The command line fills it
/// from clap's value sources; a library or binding caller leaves it empty.
/// A flag in it is consumed at its default value too, so it gets its
/// deprecation line and its series.
pub type GivenFlags = BTreeSet<String>;

/// The policy the flags name when `--policy` is left out, on every surface.
const DEFAULT_POLICY: &str = "cache_aware";

/// The positional header the consistent-hashing profile honours.
const TARGET_WORKER_HEADER: &str = "x-smg-target-worker";

/// How many requests of one key a worker carries before the next steps
/// aside, under the sticky override and the manual policy alike.
const INFLIGHT_CAP: u16 = 2;

/// The document and the report of one compilation.
#[derive(Clone, Debug, PartialEq)]
pub struct Compiled {
    /// What the flags read as: one profile per policy in use, the binding
    /// that runs it, the host inputs the flags set.
    pub spec: PolicySpec,
    /// The flags consumed and the notes that go with the document.
    pub report: Report,
}

/// What the converter did: the flags it consumed, each with the field it
/// landed in, and the notes an operator reads beside the document (what
/// the flags could not say, what a default stood in for).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Report {
    /// The consumed flags, in the order the document was built.
    pub consumed: Vec<Consumed>,
    /// Plain sentences for the operator beside the document.
    pub notes: Vec<String>,
}

impl Report {
    /// The consumed flags' ids: the series labels.
    pub fn flags(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.consumed.iter().map(|row| row.id)
    }

    /// One deprecation line per consumed flag; `spec_holds`
    /// says the lines run beside a `--policy-spec` document that was
    /// checked to agree with every flag.
    #[must_use]
    pub fn deprecation_lines(&self, removed_in: &str, spec_holds: bool) -> Vec<String> {
        self.consumed
            .iter()
            .map(|row| row.deprecation_line(removed_in, spec_holds))
            .collect()
    }
}

/// A flag the converter consumed: the document fields it landed in and the
/// value it carried there.
#[derive(Clone, Debug, PartialEq)]
pub struct Consumed {
    /// The flag's clap id, and the label of its series: `cache_threshold`.
    pub id: &'static str,
    /// The command-line spelling: `--cache-threshold`.
    pub flag: &'static str,
    /// The binding's keyword: `cache_threshold`.
    pub keyword: &'static str,
    /// The document fields the flag landed in, each with the value written
    /// there; empty for a flag nothing in the configuration reads, with
    /// [`Self::unused`] saying why.
    pub fields: Vec<Field>,
    /// The flag's value as the deprecation line shows it.
    pub shown: String,
    /// The flag's value as the document renders it.
    pub value: Value,
    /// The value the flag has when it is left out, rendered the same way;
    /// `null` for a flag with no default.
    pub default: Value,
    /// Whether the value is the default: the row exists because the flag
    /// was given.
    pub at_default: bool,
    /// Why a flag landed nowhere: the policies in use do not read it, or the
    /// mode routes nothing by it.
    pub unused: Option<String>,
}

/// One document field a flag landed in: its path
/// (`profiles.cache_aware.score.prefix-affinity.threshold`) and the value
/// written there, as the document renders it and as a line shows it. A
/// `profiles.<name>` field holds a profile's range (the forms or the stage
/// names its score may take, its picker): the policy flag that named it. A
/// null value is a presence mark: the flag implies the field is written,
/// with any value.
#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    pub path: String,
    pub value: Value,
    pub shown: String,
    pub presence: Presence,
}

/// What a row's field asks of the document: the value held, the field
/// written with any value (a structure read from its presence), or the
/// field left out (a structure the flag's policy must not take).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Presence {
    Held,
    Present,
    Absent,
}

impl Field {
    /// A presence mark: the field must be written, with any value.
    fn present(path: String) -> Self {
        Field {
            path,
            value: Value::Null,
            shown: "present".to_string(),
            presence: Presence::Present,
        }
    }

    /// An absence mark: the field must be left out.
    fn absent(path: String) -> Self {
        Field {
            path,
            value: Value::Null,
            shown: "absent".to_string(),
            presence: Presence::Absent,
        }
    }
}

impl Consumed {
    /// The start-up line: the flag, the release it leaves
    /// in, the field and value it compiled to (", which the spec already
    /// holds" under a checked `--policy-spec` document) and where to write
    /// it instead.
    #[must_use]
    pub fn deprecation_line(&self, removed_in: &str, spec_holds: bool) -> String {
        format!(
            "smg: {} is deprecated and is removed in {removed_in}; it compiled to {}; \
             write it in --policy-spec",
            self.flag,
            self.landing(spec_holds)
        )
    }

    /// The binding's form of the line: the keyword, and `policy_spec` as
    /// the place to write it.
    #[must_use]
    pub fn keyword_warning(&self, removed_in: &str) -> String {
        format!(
            "smg: the keyword {} is deprecated and is removed in {removed_in}; it compiled to \
             {}; write it in policy_spec",
            self.keyword,
            self.landing(false)
        )
    }

    /// "`<fields> = <value>`" (fields with one value grouped, fields with
    /// values of their own named each), or what became of a flag with no
    /// field.
    fn landing(&self, spec_holds: bool) -> String {
        if self.fields.is_empty() {
            let reason = self.unused.as_deref().unwrap_or("nothing reads it");
            return format!("nothing ({reason})");
        }
        let holds = if spec_holds {
            ", which the spec already holds"
        } else {
            ""
        };
        let mut groups: Vec<(&Field, Vec<&str>)> = Vec::new();
        for field in &self.fields {
            match groups
                .iter_mut()
                .find(|(head, _)| head.shown == field.shown)
            {
                Some((_, paths)) => paths.push(&field.path),
                None => groups.push((field, vec![&field.path])),
            }
        }
        let parts: Vec<String> = groups
            .iter()
            .map(|(head, paths)| {
                let paths = paths.join(" and ");
                match head.presence {
                    Presence::Held => format!("{paths} = {}", head.shown),
                    Presence::Present | Presence::Absent => format!("{paths} {}", head.shown),
                }
            })
            .collect();
        format!("{}{holds}", parts.join(" and "))
    }
}

/// A flag whose value the `--policy-spec` document does not hold: the
/// start refuses it, the document being the source of truth.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Disagreement {
    /// The command-line spelling of the flag.
    pub flag: &'static str,
    /// The flag's value as the line shows it.
    pub shown: String,
    /// The document field the flag compiled to.
    pub field: String,
    /// What the document holds there; "nothing" for a field it leaves out
    /// while the flag's value is not the default.
    pub held: String,
}

impl fmt::Display for Disagreement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} disagrees with --policy-spec {} = {}; remove the flag or change the spec \
             (the spec is the source of truth)",
            self.flag, self.shown, self.field, self.held
        )
    }
}

impl std::error::Error for Disagreement {}

/// The release that removes the routing flags, the keywords and the chart
/// values the specification replaced: the minor release after the first one
/// whose gateway reads `--policy-spec` (1.16). Every deprecation line, the
/// binding's warning and the gauge's description read it here, so a release
/// cut moves it by this one line.
pub const ROUTING_FLAGS_REMOVED_IN: &str = "1.17";

/// A routing flag with a home in the document: its clap id, its
/// command-line spelling and its binding keyword.
#[derive(Clone, Copy, Debug)]
struct Flag {
    id: &'static str,
    flag: &'static str,
    keyword: &'static str,
}

const fn flag(id: &'static str, flag: &'static str, keyword: &'static str) -> Flag {
    Flag { id, flag, keyword }
}

const POLICY: Flag = flag("policy", "--policy", "policy");
const PREFILL_POLICY: Flag = flag("prefill_policy", "--prefill-policy", "prefill_policy");
const DECODE_POLICY: Flag = flag("decode_policy", "--decode-policy", "decode_policy");
const ENCODE_POLICY: Flag = flag("encode_policy", "--encode-policy", "encode_policy");
const CACHE_THRESHOLD: Flag = flag("cache_threshold", "--cache-threshold", "cache_threshold");
const BALANCE_ABS_THRESHOLD: Flag = flag(
    "balance_abs_threshold",
    "--balance-abs-threshold",
    "balance_abs_threshold",
);
const BALANCE_REL_THRESHOLD: Flag = flag(
    "balance_rel_threshold",
    "--balance-rel-threshold",
    "balance_rel_threshold",
);
const BALANCE_PREFIX_MARGIN_TOKENS: Flag = flag(
    "balance_prefix_margin_tokens",
    "--balance-prefix-margin-tokens",
    "balance_prefix_margin_tokens",
);
const BALANCE_UNCACHED_CREDIT: Flag = flag(
    "balance_uncached_credit",
    "--balance-uncached-credit",
    "balance_uncached_credit",
);
const BALANCE_TOKEN_USAGE_THRESHOLD: Flag = flag(
    "balance_token_usage_threshold",
    "--balance-token-usage-threshold",
    "balance_token_usage_threshold",
);
const OVERLOAD_TOKEN_USAGE_THRESHOLD: Flag = flag(
    "overload_token_usage_threshold",
    "--overload-token-usage-threshold",
    "overload_token_usage_threshold",
);
const OVERLAP_DECAY: Flag = flag("overlap_decay", "--overlap-decay", "overlap_decay");
const SELECTION_TEMPERATURE: Flag = flag(
    "selection_temperature",
    "--selection-temperature",
    "selection_temperature",
);
const SELECTION_ACCOUNTING_TTL_MS: Flag = flag(
    "selection_accounting_ttl_ms",
    "--selection-accounting-ttl-ms",
    "selection_accounting_ttl_ms",
);
const EVICTION_INTERVAL: Flag = flag(
    "eviction_interval",
    "--eviction-interval",
    "eviction_interval_secs",
);
const MAX_TREE_SIZE: Flag = flag("max_tree_size", "--max-tree-size", "max_tree_size");
const BLOCK_SIZE: Flag = flag("block_size", "--block-size", "block_size");
const CACHE_BOUNDARIES: Flag = flag("cache_boundaries", "--cache-boundaries", "cache_boundaries");
const CACHE_INDEX: Flag = flag("cache_index", "--cache-index", "cache_index");
const CACHE_TTL_SECS: Flag = flag("cache_ttl_secs", "--cache-ttl-secs", "cache_ttl_secs");
const WORKER_WARMUP_SECS: Flag = flag(
    "worker_warmup_secs",
    "--worker-warmup-secs",
    "worker_warmup_secs",
);
const WORKER_WARMUP_SHARE: Flag = flag(
    "worker_warmup_share",
    "--worker-warmup-share",
    "worker_warmup_share",
);
const WORKER_WARMUP_BLOCKS: Flag = flag(
    "worker_warmup_blocks",
    "--worker-warmup-blocks",
    "worker_warmup_blocks",
);
const WORKER_WARMUP_THIN_RATIO: Flag = flag(
    "worker_warmup_thin_ratio",
    "--worker-warmup-thin-ratio",
    "worker_warmup_thin_ratio",
);
const WORKER_WARMUP_DIVERT_EVERY: Flag = flag(
    "worker_warmup_divert_every",
    "--worker-warmup-divert-every",
    "worker_warmup_divert_every",
);
const LEAST_LOAD_KV_PRESSURE_WEIGHT: Flag = flag(
    "least_load_kv_pressure_weight",
    "--least-load-kv-pressure-weight",
    "least_load_kv_pressure_weight",
);
const LEAST_LOAD_DEFAULT_THROUGHPUT: Flag = flag(
    "least_load_default_throughput",
    "--least-load-default-throughput",
    "least_load_default_throughput",
);
const LEAST_LOAD_MEAN_PREFILL_TOKENS: Flag = flag(
    "least_load_mean_prefill_tokens",
    "--least-load-mean-prefill-tokens",
    "least_load_mean_prefill_tokens",
);
const POWER_OF_TWO_LOAD_METRIC: Flag = flag(
    "power_of_two_load_metric",
    "--power-of-two-load-metric",
    "power_of_two_load_metric",
);
const BUCKET_ADJUST_INTERVAL_SECS: Flag = flag(
    "bucket_adjust_interval_secs",
    "--bucket-adjust-interval-secs",
    "bucket_adjust_interval_secs",
);
const MAX_IDLE_SECS: Flag = flag("max_idle_secs", "--max-idle-secs", "max_idle_secs");
const ASSIGNMENT_MODE: Flag = flag("assignment_mode", "--assignment-mode", "assignment_mode");
const PREFIX_TOKEN_COUNT: Flag = flag(
    "prefix_token_count",
    "--prefix-token-count",
    "prefix_token_count",
);
const PREFIX_HASH_LOAD_FACTOR: Flag = flag(
    "prefix_hash_load_factor",
    "--prefix-hash-load-factor",
    "prefix_hash_load_factor",
);
const PREFIX_HASH_BALANCE_ABS_THRESHOLD: Flag = flag(
    "prefix_hash_balance_abs_threshold",
    "--prefix-hash-balance-abs-threshold",
    "prefix_hash_balance_abs_threshold",
);
const ROUTING_KEY_OVERRIDE: Flag = flag(
    "routing_key_override",
    "--routing-key-override",
    "routing_key_override",
);
const ROUTING_KEY_HEADERS: Flag = flag(
    "routing_key_headers",
    "--routing-key-headers",
    "routing_key_headers",
);
const WORKER_OVERLOAD_PROTECTION: Flag = flag(
    "worker_overload_protection",
    "--worker-overload-protection",
    "worker_overload_protection",
);
const DISABLE_WORKER_OVERLOAD_PROTECTION: Flag = flag(
    "disable_worker_overload_protection",
    "--disable-worker-overload-protection",
    "worker_overload_protection",
);
const WORKER_OVERLOAD_WAITING_REQUESTS: Flag = flag(
    "worker_overload_waiting_requests",
    "--worker-overload-waiting-requests",
    "worker_overload_waiting_requests",
);
const WORKER_OVERLOAD_TOKEN_USAGE: Flag = flag(
    "worker_overload_token_usage",
    "--worker-overload-token-usage",
    "worker_overload_token_usage",
);
const WORKER_OVERLOAD_SHED: Flag = flag(
    "worker_overload_shed",
    "--worker-overload-shed",
    "worker_overload_shed",
);
const PD_PAIRING_MODE: Flag = flag("pd_pairing_mode", "--pd-pairing-mode", "pd_pairing_mode");

/// Every routing flag with a home in the document, for the rows of the
/// flags a configuration leaves unread.
const FLAGS: [Flag; 43] = [
    POLICY,
    PREFILL_POLICY,
    DECODE_POLICY,
    ENCODE_POLICY,
    CACHE_THRESHOLD,
    BALANCE_ABS_THRESHOLD,
    BALANCE_REL_THRESHOLD,
    BALANCE_PREFIX_MARGIN_TOKENS,
    BALANCE_UNCACHED_CREDIT,
    BALANCE_TOKEN_USAGE_THRESHOLD,
    OVERLOAD_TOKEN_USAGE_THRESHOLD,
    OVERLAP_DECAY,
    SELECTION_TEMPERATURE,
    SELECTION_ACCOUNTING_TTL_MS,
    EVICTION_INTERVAL,
    MAX_TREE_SIZE,
    BLOCK_SIZE,
    CACHE_BOUNDARIES,
    CACHE_INDEX,
    CACHE_TTL_SECS,
    WORKER_WARMUP_SECS,
    WORKER_WARMUP_SHARE,
    WORKER_WARMUP_BLOCKS,
    WORKER_WARMUP_THIN_RATIO,
    WORKER_WARMUP_DIVERT_EVERY,
    LEAST_LOAD_KV_PRESSURE_WEIGHT,
    LEAST_LOAD_DEFAULT_THROUGHPUT,
    LEAST_LOAD_MEAN_PREFILL_TOKENS,
    POWER_OF_TWO_LOAD_METRIC,
    BUCKET_ADJUST_INTERVAL_SECS,
    MAX_IDLE_SECS,
    ASSIGNMENT_MODE,
    PREFIX_TOKEN_COUNT,
    PREFIX_HASH_LOAD_FACTOR,
    PREFIX_HASH_BALANCE_ABS_THRESHOLD,
    ROUTING_KEY_OVERRIDE,
    ROUTING_KEY_HEADERS,
    WORKER_OVERLOAD_PROTECTION,
    DISABLE_WORKER_OVERLOAD_PROTECTION,
    WORKER_OVERLOAD_WAITING_REQUESTS,
    WORKER_OVERLOAD_TOKEN_USAGE,
    WORKER_OVERLOAD_SHED,
    PD_PAIRING_MODE,
];

/// A flag's value as the document renders it and as the line shows it.
trait Rendered {
    fn value(&self) -> Value;
    fn shown(&self) -> String;
}

impl Rendered for f32 {
    fn value(&self) -> Value {
        Value::from(*self)
    }
    fn shown(&self) -> String {
        self.to_string()
    }
}

impl Rendered for u32 {
    fn value(&self) -> Value {
        Value::from(*self)
    }
    fn shown(&self) -> String {
        self.to_string()
    }
}

impl Rendered for u64 {
    fn value(&self) -> Value {
        Value::from(*self)
    }
    fn shown(&self) -> String {
        self.to_string()
    }
}

impl Rendered for bool {
    fn value(&self) -> Value {
        Value::from(*self)
    }
    fn shown(&self) -> String {
        self.to_string()
    }
}

impl Rendered for &str {
    fn value(&self) -> Value {
        Value::from(*self)
    }
    fn shown(&self) -> String {
        (*self).to_string()
    }
}

impl Rendered for Vec<u32> {
    fn value(&self) -> Value {
        Value::Array(self.iter().map(|x| Value::from(*x)).collect())
    }
    fn shown(&self) -> String {
        let items: Vec<String> = self.iter().map(u32::to_string).collect();
        format!("[{}]", items.join(", "))
    }
}

impl Rendered for Vec<String> {
    fn value(&self) -> Value {
        Value::Array(self.iter().map(|x| Value::from(x.as_str())).collect())
    }
    fn shown(&self) -> String {
        format!("[{}]", self.join(", "))
    }
}

/// `value` when it differs from `default`: the written form of a stage
/// parameter.
fn off<T: PartialEq>(value: T, default: T) -> Option<T> {
    (value != default).then_some(value)
}

/// A stage with parameters when any was written, its bare name otherwise.
fn stage(written: bool, stage: ScoreStage, name: &str) -> ScoreSpec {
    if written {
        ScoreSpec::Params(stage)
    } else {
        name.into()
    }
}

fn saturating_u32<T: TryInto<u32>>(value: T) -> u32 {
    value.try_into().unwrap_or(u32::MAX)
}

fn saturating_u64<T: TryInto<u64>>(value: T) -> u64 {
    value.try_into().unwrap_or(u64::MAX)
}

/// The power-of-two profile's form under `metric`: its scorer stage, its
/// picker and the resident request tokens when the metric carries them.
fn power_of_two_form(metric: PowerOfTwoLoadMetric) -> (ScoreSpec, &'static str, Option<u32>) {
    match metric {
        PowerOfTwoLoadMetric::ExpectedWait => ("least-load".into(), "power-of-two", None),
        PowerOfTwoLoadMetric::Requests => ("least-load".into(), "power-of-two-count-first", None),
        PowerOfTwoLoadMetric::LeastRequests => ("least-load".into(), "count-first", None),
        PowerOfTwoLoadMetric::ResidentTokens { request_tokens } => {
            let request_tokens = saturating_u32(request_tokens);
            (
                ScoreSpec::Params(ScoreStage::ResidentWork(ResidentWorkParams {
                    weight: 1.0,
                    request_tokens,
                })),
                "best",
                Some(request_tokens),
            )
        }
    }
}

/// A session's idle time and sweep in the schema's seconds: a zero of
/// either legacy setting keeps keys for ever (the eviction thread off),
/// which the schema spells as its largest value.
fn lifetime(max_idle_secs: u64, eviction_interval_secs: u64) -> (u32, u32) {
    if max_idle_secs == 0 || eviction_interval_secs == 0 {
        (u32::MAX, u32::MAX)
    } else {
        (
            saturating_u32(max_idle_secs),
            saturating_u32(eviction_interval_secs),
        )
    }
}

fn assignment_of(mode: ManualAssignmentMode) -> Assignment {
    match mode {
        ManualAssignmentMode::Random => Assignment::Random,
        ManualAssignmentMode::MinLoad => Assignment::MinLoad,
        ManualAssignmentMode::MinGroup => Assignment::MinGroup,
        ManualAssignmentMode::Delegate => Assignment::Delegate,
    }
}

fn assignment_name(mode: ManualAssignmentMode) -> &'static str {
    match mode {
        ManualAssignmentMode::Random => "random",
        ManualAssignmentMode::MinLoad => "min_load",
        ManualAssignmentMode::MinGroup => "min_group",
        ManualAssignmentMode::Delegate => "delegate",
    }
}

/// Compile `config`'s routing flags into the document and the report with
/// no flag known to have been given: a flag is consumed where its value
/// differs from its default, and the policy names where a mode reads them.
#[must_use]
pub fn compile(config: &RouterConfig) -> Compiled {
    compile_given(config, &GivenFlags::new())
}

/// [`compile`] with the flags the front end saw on its command line: one
/// of them is consumed at its default value too, so a given flag always
/// gets its line and its series.
#[must_use]
pub fn compile_given(config: &RouterConfig, given: &GivenFlags) -> Compiled {
    Compiler {
        config,
        given,
        view: ViewSpec::default(),
        index: None,
        plans: PlansSpec::default(),
        profiles: BTreeMap::new(),
        binding: BindingSpec::default(),
        manual: None,
        consumed: Vec::new(),
        notes: Vec::new(),
        implied: Vec::new(),
    }
    .run()
}

/// The binding keyword of every routing flag, for a front end that pins
/// its own keyword set to the converter's.
pub fn keywords() -> impl Iterator<Item = &'static str> {
    FLAGS.iter().map(|flag| flag.keyword)
}

/// Every routing flag's id, the names a given set is made of.
pub fn flag_ids() -> impl Iterator<Item = &'static str> {
    FLAGS.iter().map(|flag| flag.id)
}

/// The ids of the routing flags behind a binding keyword, for a front end
/// that marks its given set by keyword.
pub fn flag_ids_of(keyword: &str) -> impl Iterator<Item = &'static str> + '_ {
    FLAGS
        .iter()
        .filter(move |flag| flag.keyword == keyword)
        .map(|flag| flag.id)
}

/// The side of a plan a profile serves: the pair sides insert their
/// capacity filter before `not-tried`, as the engine's pair presets do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    Single,
    Prefill,
    Decode,
    Encode,
}

impl Side {
    fn filters(self, own: &[&str]) -> Vec<String> {
        let mut filters: Vec<String> = own.iter().map(|f| (*f).to_string()).collect();
        let extra = match self {
            Self::Prefill => Some("has-capacity"),
            Self::Decode => Some("has-room"),
            Self::Single | Self::Encode => None,
        };
        if let Some(extra) = extra {
            let at = filters.len().min(2);
            filters.insert(at, extra.to_string());
        }
        filters
    }
}

/// The filters of a policy that picks among every usable worker.
const STANDARD: [&str; 3] = ["healthy", "not-overloaded", "not-tried"];
/// The filters of a policy that places by key or bucket and never retries
/// elsewhere on its own.
const KEYED: [&str; 2] = ["healthy", "not-overloaded"];

const LABEL_NOTE: &str = "worker labels `policy` are not visible to the converter; under the spec an \
                          explicit binding wins over a label (today the label wins): add a binding per \
                          labelled model";
const PER_WORKER_OVERLOAD_NOTE: &str =
    "per-worker overload blocks of registered workers resolve over \
                                        these defaults at run time; the spec cannot carry them";

/// The cache-aware knobs of one policy configuration, as the stages read
/// them.
#[derive(Clone, Copy)]
struct CacheKnobs {
    threshold: f32,
    abs: usize,
    rel: f32,
    margin: u32,
    credit: bool,
    sweep: u64,
    max_size: usize,
    block: usize,
    spread: f32,
    ceiling: f32,
    decay: f32,
    temperature: f32,
    index: CacheIndexKind,
    ttl: u64,
    accounting_ttl_ms: u64,
}

/// One compilation: the document's parts as they fill, the rows as they
/// are consumed.
struct Compiler<'a> {
    config: &'a RouterConfig,
    given: &'a GivenFlags,
    view: ViewSpec,
    index: Option<IndexSpec>,
    plans: PlansSpec,
    profiles: BTreeMap<String, ProfileSpec>,
    binding: BindingSpec,
    /// The manual policy's sessions block, when its profile was built; the
    /// sticky override's block takes its place when the override is on.
    manual: Option<SessionsSpec>,
    consumed: Vec<Consumed>,
    notes: Vec<String>,
    /// Fields the policy flag implies through the structure it selects,
    /// gathered by the builders while their knob rows are at default and
    /// taken by their own profile's policy row.
    implied: Vec<Field>,
}

impl Compiler<'_> {
    fn run(mut self) -> Compiled {
        let config = self.config;
        match &config.mode {
            RoutingMode::PrefillDecode {
                prefill_policy,
                decode_policy,
                ..
            } => self.pair(
                prefill_policy.as_ref(),
                decode_policy.as_ref(),
                None,
                "--pd-disaggregation",
            ),
            RoutingMode::EncodePrefillDecode {
                encode_policy,
                prefill_policy,
                decode_policy,
                ..
            } => self.pair(
                prefill_policy.as_ref(),
                decode_policy.as_ref(),
                Some(encode_policy.as_ref()),
                "--epd-disaggregation",
            ),
            RoutingMode::Regular { .. }
            | RoutingMode::OpenAI { .. }
            | RoutingMode::Anthropic { .. }
            | RoutingMode::Gemini { .. } => self.single(),
        }
        self.sessions();
        self.overload();
        self.boundaries();
        self.unused_given();
        if config.discovery.is_some() {
            self.notes.push(LABEL_NOTE.to_string());
        }
        let Compiler {
            view,
            index,
            plans,
            profiles,
            binding,
            consumed,
            notes,
            implied,
            ..
        } = self;
        debug_assert!(
            implied.is_empty(),
            "implied fields without a profile row: {implied:?}"
        );
        let mut spec = PolicySpec::new(profiles);
        spec.generation = 1;
        spec.view = view;
        if let Some(index) = index {
            spec.index.insert("*".to_string(), index);
        }
        spec.plans = plans;
        spec.bindings = vec![binding];
        Compiled {
            spec,
            report: Report { consumed, notes },
        }
    }

    /// A row for every routing flag the front end saw that nothing in this
    /// configuration read: it gets its line and its series like the rest,
    /// and the reason instead of a field.
    fn unused_given(&mut self) {
        let config = self.config;
        let reason = match &config.mode {
            RoutingMode::PrefillDecode { .. } => {
                "the prefill and decode policies do not read it".to_string()
            }
            RoutingMode::EncodePrefillDecode { .. } => {
                "the encode, prefill and decode policies do not read it".to_string()
            }
            _ => format!("the {} policy does not read it", config.policy.name()),
        };
        // A keyword two flags share (the overload switch and its disable
        // form) is read once: a front end that marks by keyword gives both.
        for flag in FLAGS {
            let read = self.consumed.iter().any(|row| row.keyword == flag.keyword);
            if self.given.contains(flag.id) && !read {
                self.unused_row(flag, String::new(), reason.clone());
            }
        }
    }

    /// The regular modes: one profile named after the policy, bound to
    /// every model.
    fn single(&mut self) {
        let config = self.config;
        let policy = &config.policy;
        let name = policy.name();
        let profile = self.profile(policy, Side::Single, name);
        self.binding.profile = Some(name.to_string());
        self.profile_row(
            POLICY,
            "bindings[0].profile",
            name,
            &profile,
            policy,
            Some(DEFAULT_POLICY),
        );
        self.profiles.insert(name.to_string(), profile);
        if name == DEFAULT_POLICY && !self.given.contains(POLICY.id) {
            self.notes.push(format!(
                "no --policy given: the default profile {name} was written"
            ));
        }
    }

    /// The disaggregated modes: a profile per side under the `pd-pair` or
    /// `epd` plan, each side's policy or the main policy as its fallback,
    /// the encode side's default written out.
    fn pair(
        &mut self,
        prefill: Option<&PolicyConfig>,
        decode: Option<&PolicyConfig>,
        epd: Option<Option<&PolicyConfig>>,
        mode_flag: &str,
    ) {
        let config = self.config;
        let main = &config.policy;
        let prefill_profile = self.profile(prefill.unwrap_or(main), Side::Prefill, "prefill");
        let decode_profile = self.profile(decode.unwrap_or(main), Side::Decode, "decode");
        let mut sides = BTreeMap::new();
        sides.insert("prefill".to_string(), "prefill".to_string());
        sides.insert("decode".to_string(), "decode".to_string());
        let mut phases = vec![Phase::Prefill, Phase::Decode];
        let mut plan = PlanName::PdPair;
        if let Some(encode) = epd {
            let policy = encode.unwrap_or(&DEFAULT_ENCODE_POLICY);
            let encode_profile = self.profile(policy, Side::Encode, "encode");
            sides.insert("encode".to_string(), "encode".to_string());
            phases.insert(0, Phase::Encode);
            plan = PlanName::Epd;
            match encode {
                Some(policy) => self.profile_row(
                    ENCODE_POLICY,
                    "bindings[0].profiles.encode",
                    "encode",
                    &encode_profile,
                    policy,
                    None,
                ),
                None => {
                    // No flag, no claims: the default profile's marks go.
                    self.claims("encode");
                    self.notes.push(format!(
                        "no --encode-policy given: the default encode profile {} was written",
                        DEFAULT_ENCODE_POLICY.name()
                    ));
                }
            }
            self.profiles.insert("encode".to_string(), encode_profile);
        }
        match prefill {
            Some(policy) => self.profile_row(
                PREFILL_POLICY,
                "bindings[0].profiles.prefill",
                "prefill",
                &prefill_profile,
                policy,
                None,
            ),
            None => self.profile_row(
                POLICY,
                "bindings[0].profiles.prefill",
                "prefill",
                &prefill_profile,
                main,
                Some(DEFAULT_POLICY),
            ),
        }
        match decode {
            Some(policy) => self.profile_row(
                DECODE_POLICY,
                "bindings[0].profiles.decode",
                "decode",
                &decode_profile,
                policy,
                None,
            ),
            None => self.profile_row(
                POLICY,
                "bindings[0].profiles.decode",
                "decode",
                &decode_profile,
                main,
                Some(DEFAULT_POLICY),
            ),
        }
        if prefill.is_some() && decode.is_some() {
            self.notes.push(format!(
                "the main policy {} routes no request under {mode_flag}: it compiled to nothing",
                main.name()
            ));
            if main.name() != DEFAULT_POLICY || self.given.contains(POLICY.id) {
                self.unused_row(
                    POLICY,
                    main.name().to_string(),
                    format!("{} routes no request under {mode_flag}", main.name()),
                );
            }
        }
        self.profiles.insert("prefill".to_string(), prefill_profile);
        self.profiles.insert("decode".to_string(), decode_profile);
        self.binding.plan = Some(plan);
        self.binding.profiles = sides;
        self.binding.matcher.phase = phases;
        self.plans.pd_pair.pairing = match config.pd_pairing_mode {
            PdPairingMode::Off => Pairing::Off,
            PdPairingMode::Lenient => Pairing::Lenient,
            PdPairingMode::Strict => Pairing::Strict,
        };
        self.row(
            PD_PAIRING_MODE,
            vec!["plans.pd-pair.pairing".to_string()],
            &config.pd_pairing_mode.as_str(),
            Some(&"lenient"),
        );
    }

    /// A policy's profile: its expansion with the parameters
    /// its flags set, each flag consumed into its stage's field.
    fn profile(&mut self, policy: &PolicyConfig, side: Side, name: &str) -> ProfileSpec {
        let p = format!("profiles.{name}");
        match policy {
            PolicyConfig::Random => ProfileSpec {
                filter: side.filters(&STANDARD),
                score: vec![],
                pick: "random".into(),
            },
            PolicyConfig::RoundRobin => ProfileSpec {
                filter: side.filters(&STANDARD),
                score: vec![],
                pick: "round-robin".into(),
            },
            PolicyConfig::Passthrough => ProfileSpec {
                filter: side.filters(&STANDARD),
                score: vec![],
                pick: "passthrough".into(),
            },
            PolicyConfig::LeastLoad {
                kv_pressure_weight,
                mean_prefill_tokens,
                default_throughput,
            } => {
                let least_load = self.least_load(
                    &p,
                    *kv_pressure_weight as f32,
                    *mean_prefill_tokens,
                    *default_throughput as f32,
                );
                ProfileSpec {
                    filter: side.filters(&STANDARD),
                    score: vec![least_load],
                    pick: "best".into(),
                }
            }
            PolicyConfig::PowerOfTwo { load_metric } => self.power_of_two(&p, side, *load_metric),
            PolicyConfig::CacheAware {
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                balance_prefix_margin_tokens,
                balance_uncached_credit,
                eviction_interval_secs,
                max_tree_size,
                block_size,
                balance_token_usage_threshold,
                overload_token_usage_threshold,
                overlap_decay,
                selection_temperature,
                cache_index,
                cache_ttl_secs,
                selection_accounting_ttl_ms,
                ..
            } => self.cache_aware(
                &p,
                side,
                CacheKnobs {
                    threshold: *cache_threshold,
                    abs: *balance_abs_threshold,
                    rel: *balance_rel_threshold,
                    margin: *balance_prefix_margin_tokens,
                    credit: *balance_uncached_credit,
                    sweep: *eviction_interval_secs,
                    max_size: *max_tree_size,
                    block: *block_size,
                    spread: *balance_token_usage_threshold,
                    ceiling: *overload_token_usage_threshold,
                    decay: *overlap_decay,
                    temperature: *selection_temperature,
                    index: *cache_index,
                    ttl: *cache_ttl_secs,
                    accounting_ttl_ms: *selection_accounting_ttl_ms,
                },
            ),
            PolicyConfig::Bucket {
                balance_abs_threshold,
                balance_rel_threshold,
                bucket_adjust_interval_secs,
            } => self.bucket(
                &p,
                side,
                *balance_abs_threshold,
                *balance_rel_threshold,
                *bucket_adjust_interval_secs,
            ),
            PolicyConfig::Manual {
                eviction_interval_secs,
                max_idle_secs,
                assignment_mode,
            } => {
                self.manual_sessions(*eviction_interval_secs, *max_idle_secs, *assignment_mode);
                ProfileSpec {
                    filter: side.filters(&KEYED),
                    score: vec![],
                    pick: "manual".into(),
                }
            }
            PolicyConfig::ConsistentHashing => {
                // The encode pool's ring keys by item, never by a header.
                if side != Side::Encode {
                    self.header(MANUAL_SESSION_HEADER);
                    self.header(TARGET_WORKER_HEADER);
                }
                // The ring keys by the session, unbounded: a prefix key or a
                // bound written under the name would route another way.
                let at = |field: &str| format!("{p}.score.consistent-hash.{field}");
                self.implied.push(Field::absent(at("prefix_tokens")));
                self.implied.push(Field::absent(at("bounded_load")));
                ProfileSpec {
                    filter: side.filters(&STANDARD),
                    score: vec!["consistent-hash".into()],
                    pick: "best".into(),
                }
            }
            PolicyConfig::PrefixHash {
                prefix_token_count,
                load_factor,
                balance_abs_threshold,
                ..
            } => self.prefix_hash(
                &p,
                side,
                *prefix_token_count,
                *load_factor as f32,
                *balance_abs_threshold,
            ),
        }
    }

    /// The least-load scorer with the three tunables, written off their
    /// defaults.
    fn least_load(
        &mut self,
        p: &str,
        kv_pressure_weight: f32,
        mean_prefill_tokens: u32,
        default_throughput: f32,
    ) -> ScoreSpec {
        let at = |field: &str| format!("{p}.score.least-load.{field}");
        self.row(
            LEAST_LOAD_KV_PRESSURE_WEIGHT,
            vec![at("kv_pressure_weight")],
            &kv_pressure_weight,
            Some(&0.15_f32),
        );
        self.row(
            LEAST_LOAD_MEAN_PREFILL_TOKENS,
            vec![at("mean_prefill_tokens")],
            &mean_prefill_tokens,
            Some(&1024_u32),
        );
        self.row(
            LEAST_LOAD_DEFAULT_THROUGHPUT,
            vec![at("default_throughput")],
            &default_throughput,
            Some(&2000.0_f32),
        );
        let params = LeastLoadParams {
            weight: 1.0,
            kv_pressure_weight: off(kv_pressure_weight, 0.15),
            mean_prefill_tokens: off(mean_prefill_tokens, 1024),
            default_throughput: off(default_throughput, 2000.0),
        };
        stage(
            params != LeastLoadParams::default(),
            ScoreStage::LeastLoad(params),
            "least-load",
        )
    }

    /// The four forms of `power_of_two` by its load metric: two draws
    /// compared by the expected wait; the two request-count forms through
    /// the engine's count-first pickers (the fewest in flight, then the
    /// expected wait); the resident-tokens form over every candidate.
    fn power_of_two(&mut self, p: &str, side: Side, metric: PowerOfTwoLoadMetric) -> ProfileSpec {
        let typed = metric.spell();
        let (score, pick, resident) = power_of_two_form(metric);
        self.row_shown(
            POWER_OF_TWO_LOAD_METRIC,
            vec![format!("{p}.pick")],
            &pick,
            Some(&"power-of-two"),
            typed.clone(),
        );
        let stage = stage_name(&serde_json::to_value(&score).unwrap_or(Value::Null));
        let stage = stage.as_str().unwrap_or_default().to_string();
        self.row_shown(
            POWER_OF_TWO_LOAD_METRIC,
            vec![format!("{p}.score")],
            &vec![stage],
            Some(&vec!["least-load".to_string()]),
            typed.clone(),
        );
        if let Some(request_tokens) = resident {
            self.row_shown(
                POWER_OF_TWO_LOAD_METRIC,
                vec![format!("{p}.score.resident-work.request_tokens")],
                &request_tokens,
                None,
                typed,
            );
        }
        ProfileSpec {
            filter: side.filters(&STANDARD),
            score: vec![score],
            pick: pick.into(),
        }
    }

    /// The cache-aware expansion: prefix affinity with its match threshold
    /// and spill gate, KV pressure, the expected wait and the warm-up
    /// slice, picked by the best score or a temperature over it; the index
    /// settings go to the per-model index map and the block size to the
    /// view.
    fn cache_aware(&mut self, p: &str, side: Side, k: CacheKnobs) -> ProfileSpec {
        let affinity_at = |field: &str| format!("{p}.score.prefix-affinity.{field}");
        self.row(
            CACHE_THRESHOLD,
            vec![affinity_at("threshold")],
            &k.threshold,
            Some(&0.3_f32),
        );
        self.row(
            OVERLAP_DECAY,
            vec![affinity_at("decay")],
            &k.decay,
            Some(&0.0_f32),
        );
        self.row(
            SELECTION_ACCOUNTING_TTL_MS,
            vec![affinity_at("accounting_ttl_ms")],
            &k.accounting_ttl_ms,
            Some(&0_u64),
        );
        let abs = saturating_u32(k.abs);
        self.row(
            BALANCE_ABS_THRESHOLD,
            vec![affinity_at("spill.abs")],
            &abs,
            Some(&64_u32),
        );
        self.row(
            BALANCE_REL_THRESHOLD,
            vec![affinity_at("spill.rel")],
            &k.rel,
            Some(&1.5_f32),
        );
        self.row(
            BALANCE_PREFIX_MARGIN_TOKENS,
            vec![affinity_at("spill.prefix_margin_tokens")],
            &k.margin,
            Some(&0_u32),
        );
        self.row(
            BALANCE_UNCACHED_CREDIT,
            vec![affinity_at("spill.uncached_credit")],
            &k.credit,
            Some(&false),
        );
        let spill = SpillParams {
            abs: off(abs, 64),
            rel: off(k.rel, 1.5),
            prefix_margin_tokens: off(k.margin, 0),
            uncached_credit: off(k.credit, false),
        };
        let affinity = PrefixAffinityParams {
            weight: 1.0,
            threshold: off(k.threshold, 0.3),
            decay: off(k.decay, 0.0),
            source: None,
            accounting_ttl_ms: off(k.accounting_ttl_ms, 0),
            spill: (spill != SpillParams::default()).then_some(spill),
        };
        let pressure_at = |field: &str| format!("{p}.score.kv-pressure.{field}");
        self.row(
            BALANCE_TOKEN_USAGE_THRESHOLD,
            vec![pressure_at("spread")],
            &k.spread,
            Some(&1.0_f32),
        );
        self.row(
            OVERLOAD_TOKEN_USAGE_THRESHOLD,
            vec![pressure_at("ceiling")],
            &k.ceiling,
            Some(&1.0_f32),
        );
        let pressure = KvPressureParams {
            weight: 1.0,
            spread: off(k.spread, 1.0),
            ceiling: off(k.ceiling, 1.0),
        };
        let warmup = self.warmup(p);
        let score = vec![
            stage(
                affinity != PrefixAffinityParams::default(),
                ScoreStage::PrefixAffinity(affinity),
                "prefix-affinity",
            ),
            stage(
                pressure != KvPressureParams::default(),
                ScoreStage::KvPressure(pressure),
                "kv-pressure",
            ),
            "least-load".into(),
            warmup,
        ];
        self.row(
            SELECTION_TEMPERATURE,
            vec![format!("{p}.pick.best-with-temperature.temperature")],
            &k.temperature,
            Some(&0.0_f32),
        );
        let pick = if k.temperature > 0.0 {
            PickSpec::Params(PickStage::BestWithTemperature(BestWithTemperatureParams {
                temperature: k.temperature,
            }))
        } else {
            "best".into()
        };
        self.index_settings(k.index, k.ttl, k.max_size, k.sweep);
        let block = saturating_u32(k.block);
        self.row(
            BLOCK_SIZE,
            vec!["view.block_size".to_string()],
            &block,
            Some(&16_u32),
        );
        if block != 16 {
            self.view.block_size = Some(block);
        }
        ProfileSpec {
            filter: side.filters(&STANDARD),
            score,
            pick,
        }
    }

    /// The warm-up slice from the gateway's five warm-up settings, which
    /// only the cache-aware profile reads.
    fn warmup(&mut self, p: &str) -> ScoreSpec {
        let config = self.config;
        let at = |field: &str| format!("{p}.score.warmup-slice.{field}");
        let blocks = saturating_u64(config.worker_warmup_blocks);
        self.row(
            WORKER_WARMUP_SECS,
            vec![at("secs")],
            &config.worker_warmup_secs,
            Some(&60_u64),
        );
        self.row(
            WORKER_WARMUP_SHARE,
            vec![at("share")],
            &config.worker_warmup_share,
            Some(&0.25_f32),
        );
        self.row(
            WORKER_WARMUP_BLOCKS,
            vec![at("blocks")],
            &blocks,
            Some(&1024_u64),
        );
        self.row(
            WORKER_WARMUP_THIN_RATIO,
            vec![at("thin_ratio")],
            &config.worker_warmup_thin_ratio,
            Some(&0.5_f32),
        );
        self.row(
            WORKER_WARMUP_DIVERT_EVERY,
            vec![at("divert_every")],
            &config.worker_warmup_divert_every,
            Some(&8_u64),
        );
        let params = WarmupSliceParams {
            weight: 1.0,
            secs: off(config.worker_warmup_secs, 60),
            share: off(config.worker_warmup_share, 0.25),
            blocks: off(blocks, 1024),
            thin_ratio: off(config.worker_warmup_thin_ratio, 0.5),
            divert_every: off(config.worker_warmup_divert_every, 8),
        };
        stage(
            params != WarmupSliceParams::default(),
            ScoreStage::WarmupSlice(params),
            "warmup-slice",
        )
    }

    /// The prefix index's settings, per model: one entry for every model
    /// (the flags know no model), written when any setting is off its
    /// default.
    fn index_settings(&mut self, kind: CacheIndexKind, ttl: u64, max_size: usize, sweep: u64) {
        let at = |field: &str| format!("index[\"*\"].{field}");
        let (kind_name, kind) = match kind {
            CacheIndexKind::Tree => ("tree", IndexKind::Tree),
            CacheIndexKind::Hash => ("hash", IndexKind::Hash),
        };
        let settings = IndexSpec {
            kind,
            ttl_s: saturating_u32(ttl),
            max_size: saturating_u64(max_size),
            sweep_s: saturating_u32(sweep),
        };
        self.row(CACHE_INDEX, vec![at("kind")], &kind_name, Some(&"tree"));
        self.row(
            CACHE_TTL_SECS,
            vec![at("ttl_s")],
            &settings.ttl_s,
            Some(&180_u32),
        );
        self.row(
            MAX_TREE_SIZE,
            vec![at("max_size")],
            &settings.max_size,
            Some(&67_108_864_u64),
        );
        self.row(
            EVICTION_INTERVAL,
            vec![at("sweep_s")],
            &settings.sweep_s,
            Some(&120_u32),
        );
        if settings == IndexSpec::default() {
            return;
        }
        match &self.index {
            None => self.index = Some(settings),
            Some(first) if *first != settings => self.notes.push(
                "two cache-aware profiles set different index settings; the first profile's \
                 were written (the index is per model)"
                    .to_string(),
            ),
            Some(_) => {}
        }
    }

    /// The bucket scorer over the two balance thresholds and its cadence.
    fn bucket(
        &mut self,
        p: &str,
        side: Side,
        abs: usize,
        rel: f32,
        interval: usize,
    ) -> ProfileSpec {
        let at = |field: &str| format!("{p}.score.bucket.{field}");
        let abs = saturating_u32(abs);
        let interval = saturating_u32(interval);
        self.row(
            BALANCE_ABS_THRESHOLD,
            vec![at("abs_threshold")],
            &abs,
            Some(&64_u32),
        );
        self.row(
            BALANCE_REL_THRESHOLD,
            vec![at("rel_threshold")],
            &rel,
            Some(&1.5_f32),
        );
        self.row(
            BUCKET_ADJUST_INTERVAL_SECS,
            vec![at("adjust_interval_s")],
            &interval,
            Some(&5_u32),
        );
        let params = BucketParams {
            weight: 1.0,
            abs_threshold: off(abs, 64),
            rel_threshold: off(rel, 1.5),
            adjust_interval_s: off(interval, 5),
        };
        ProfileSpec {
            filter: side.filters(&KEYED),
            score: vec![stage(
                params != BucketParams::default(),
                ScoreStage::Bucket(params),
                "bucket",
            )],
            pick: "best".into(),
        }
    }

    /// The prefix-keyed ring with its bounded-load rule.
    fn prefix_hash(
        &mut self,
        p: &str,
        side: Side,
        prefix_tokens: usize,
        factor: f32,
        margin: usize,
    ) -> ProfileSpec {
        let at = |field: &str| format!("{p}.score.consistent-hash.{field}");
        let prefix_tokens = saturating_u32(prefix_tokens);
        let margin = saturating_u32(margin);
        self.row(
            PREFIX_TOKEN_COUNT,
            vec![at("prefix_tokens")],
            &prefix_tokens,
            Some(&256_u32),
        );
        self.row(
            PREFIX_HASH_LOAD_FACTOR,
            vec![at("bounded_load.factor")],
            &factor,
            Some(&1.25_f32),
        );
        self.row(
            PREFIX_HASH_BALANCE_ABS_THRESHOLD,
            vec![at("bounded_load.abs_margin")],
            &margin,
            Some(&10_u32),
        );
        // The prefix form is read from the presence of `prefix_tokens` and
        // the bound from the presence of its block (an empty block is the
        // bound with its defaults), so the policy flag marks both present
        // whatever the knobs say; the knob rows hold the values.
        self.implied.push(Field::present(at("prefix_tokens")));
        self.implied.push(Field::present(at("bounded_load")));
        // Written whole: the engine reads the prefix form from the presence
        // of `prefix_tokens` and the bound from the presence of its block,
        // so the defaults are the meaning here, not an omission.
        let params = ConsistentHashParams {
            weight: 1.0,
            prefix_tokens: Some(prefix_tokens),
            bounded_load: Some(BoundedLoadParams {
                factor: Some(factor),
                abs_margin: Some(margin),
            }),
        };
        ProfileSpec {
            filter: side.filters(&STANDARD),
            score: vec![ScoreSpec::Params(ScoreStage::ConsistentHash(params))],
            pick: "best".into(),
        }
    }

    /// The manual policy's sessions: its key header, its idle time and
    /// sweep, how a first-seen key is placed, the cap of two in flight.
    fn manual_sessions(&mut self, sweep: u64, idle: u64, mode: ManualAssignmentMode) {
        let (ttl_s, sweep_s) = lifetime(idle, sweep);
        self.lifetime_rows(idle, sweep, ttl_s, sweep_s);
        self.row(
            ASSIGNMENT_MODE,
            vec!["bindings[0].sessions.assignment".to_string()],
            &assignment_name(mode),
            Some(&"random"),
        );
        self.manual.get_or_insert_with(|| SessionsSpec {
            sticky_header: MANUAL_SESSION_HEADER.to_string(),
            headers: vec![MANUAL_SESSION_HEADER.to_string()],
            ttl_s,
            sweep_s,
            assignment: assignment_of(mode),
            inflight_cap: INFLIGHT_CAP,
            issue_session_id: false,
            recovery: Recovery::Original,
        });
    }

    /// The idle-time and sweep rows of a sessions block. A zero idle time
    /// keeps keys for ever and writes the sweep as for ever too, so the idle
    /// flag owns both fields then, and a sweep given beside it has its line
    /// with the reason; a zero sweep does the same the other way round. Off
    /// zero, each flag owns its field.
    fn lifetime_rows(&mut self, idle: u64, sweep: u64, ttl_s: u32, sweep_s: u32) {
        let ttl = "bindings[0].sessions.ttl_s".to_string();
        let sweep_field = "bindings[0].sessions.sweep_s".to_string();
        if idle == 0 {
            self.row_shown(
                MAX_IDLE_SECS,
                vec![ttl, sweep_field.clone()],
                &ttl_s,
                Some(&14_400_u32),
                idle.to_string(),
            );
            if sweep == 0 {
                self.row_shown(
                    EVICTION_INTERVAL,
                    vec![sweep_field],
                    &sweep_s,
                    Some(&120_u32),
                    sweep.to_string(),
                );
            } else if self.given.contains(EVICTION_INTERVAL.id) || sweep != 120 {
                self.unused_row(
                    EVICTION_INTERVAL,
                    sweep.to_string(),
                    "a zero idle time keeps keys for ever, so the sweep is for ever too"
                        .to_string(),
                );
            }
        } else if sweep == 0 {
            self.row_shown(
                EVICTION_INTERVAL,
                vec![sweep_field, ttl],
                &sweep_s,
                Some(&120_u32),
                sweep.to_string(),
            );
            if self.given.contains(MAX_IDLE_SECS.id) || idle != 14_400 {
                self.unused_row(
                    MAX_IDLE_SECS,
                    idle.to_string(),
                    "a zero sweep keeps keys for ever, so the idle time is for ever too"
                        .to_string(),
                );
            }
        } else {
            self.row(MAX_IDLE_SECS, vec![ttl], &ttl_s, Some(&14_400_u32));
            self.row(
                EVICTION_INTERVAL,
                vec![sweep_field],
                &sweep_s,
                Some(&120_u32),
            );
        }
    }

    /// The binding's sessions block: the sticky override's when it is on,
    /// the manual policy's otherwise; its headers join the view.
    fn sessions(&mut self) {
        let config = self.config;
        let sticky = &config.routing_key_override;
        let block = if sticky.enabled {
            Some(self.override_sessions(sticky))
        } else {
            self.manual.take()
        };
        if let Some(block) = block {
            for header in block.headers.clone() {
                self.header(&header);
            }
            self.binding.sessions = Some(block);
        }
    }

    /// The sticky override's sessions: its ordered key headers (the first
    /// the sticky header), its idle time and sweep, how a first-seen key is
    /// placed, the cap of two in flight; under `delegate` the latest pin
    /// leads when both of a key's pins are back, under the other
    /// assignments the first, as the legacy override does.
    fn override_sessions(&mut self, sticky: &RoutingKeyOverrideConfig) -> SessionsSpec {
        let (ttl_s, sweep_s) = lifetime(sticky.max_idle_secs, sticky.eviction_interval_secs);
        let mut headers: Vec<String> = Vec::with_capacity(sticky.headers.len());
        for header in &sticky.headers {
            let name = header.to_ascii_lowercase();
            if !headers.contains(&name) {
                headers.push(name);
            }
        }
        if headers.is_empty() {
            headers.push(MANUAL_SESSION_HEADER.to_string());
        }
        let sticky_header = headers[0].clone();
        self.row(
            ROUTING_KEY_OVERRIDE,
            vec!["bindings[0].sessions".to_string()],
            &true,
            Some(&false),
        );
        self.row(
            ROUTING_KEY_HEADERS,
            vec!["bindings[0].sessions.headers".to_string()],
            &headers,
            Some(&vec![MANUAL_SESSION_HEADER.to_string()]),
        );
        self.lifetime_rows(
            sticky.max_idle_secs,
            sticky.eviction_interval_secs,
            ttl_s,
            sweep_s,
        );
        self.row(
            ASSIGNMENT_MODE,
            vec!["bindings[0].sessions.assignment".to_string()],
            &assignment_name(sticky.assignment_mode),
            Some(&"delegate"),
        );
        let recovery = if sticky.assignment_mode == ManualAssignmentMode::Delegate {
            Recovery::Latest
        } else {
            Recovery::Original
        };
        SessionsSpec {
            sticky_header,
            headers,
            ttl_s,
            sweep_s,
            assignment: assignment_of(sticky.assignment_mode),
            inflight_cap: INFLIGHT_CAP,
            issue_session_id: false,
            recovery,
        }
    }

    /// A request header a stage reads, listed once in the view.
    fn header(&mut self, name: &str) {
        if !self.view.headers.iter().any(|h| h == name) {
            self.view.headers.push(name.to_string());
        }
    }

    /// The overload veto's defaults on the binding: the switch, the two
    /// thresholds and the shed rule, the block written when any is off its
    /// default. A threshold switched off has no spec form and is noted.
    fn overload(&mut self) {
        let config = self.config;
        let at = |field: &str| format!("bindings[0].overload.{field}");
        if config.worker_overload_protection {
            self.row(
                WORKER_OVERLOAD_PROTECTION,
                vec![at("enabled")],
                &true,
                Some(&true),
            );
        } else {
            self.row(
                DISABLE_WORKER_OVERLOAD_PROTECTION,
                vec![at("enabled")],
                &false,
                Some(&true),
            );
        }
        let waiting = config.worker_overload_waiting_requests.map(saturating_u32);
        match waiting {
            Some(waiting) => self.row(
                WORKER_OVERLOAD_WAITING_REQUESTS,
                vec![at("waiting_requests")],
                &waiting,
                Some(&8_u32),
            ),
            None => self.notes.push(
                "the waiting-requests overload signal is off; the spec has no off value: write \
                 a threshold no worker reaches"
                    .to_string(),
            ),
        }
        let usage = config.worker_overload_token_usage.map(|usage| usage as f32);
        match usage {
            Some(usage) => self.row(
                WORKER_OVERLOAD_TOKEN_USAGE,
                vec![at("token_usage")],
                &usage,
                Some(&0.8_f32),
            ),
            None => self.notes.push(
                "the token-usage overload signal is off; the spec has no off value: write a \
                 ceiling no worker reaches"
                    .to_string(),
            ),
        }
        self.row(
            WORKER_OVERLOAD_SHED,
            vec![at("shed_when_all_vetoed")],
            &config.worker_overload_shed,
            Some(&false),
        );
        let block = OverloadSpec {
            enabled: config.worker_overload_protection,
            waiting_requests: waiting.and_then(|waiting| off(waiting, 8)),
            token_usage: usage.and_then(|usage| off(usage, 0.8)),
            shed_when_all_vetoed: config.worker_overload_shed,
        };
        if block != OverloadSpec::default() {
            self.binding.overload = Some(block);
            self.notes.push(PER_WORKER_OVERLOAD_NOTE.to_string());
        }
    }

    /// The hash index's and the prefix ring's boundary positions: a host
    /// input of the view.
    fn boundaries(&mut self) {
        let boundaries: Vec<u32> = self
            .config
            .cache_boundaries
            .iter()
            .map(|&position| saturating_u32(position))
            .collect();
        self.row(
            CACHE_BOUNDARIES,
            vec!["view.boundaries".to_string()],
            &boundaries,
            Some(&Vec::<u32>::new()),
        );
        self.view.boundaries = boundaries;
    }

    /// A flag at its default is consumed only when the front end saw it:
    /// a value the operator wrote is the operator's, default or not; one
    /// nobody wrote has no line.
    fn skips(&self, flag: &Flag, at_default: bool) -> bool {
        at_default && !self.given.contains(flag.id)
    }

    /// Consume `flag` into `fields` with `value`: a flag at its default is
    /// consumed only when the front end saw it; a flag seen twice (one
    /// value landing in two fields) keeps one row with both fields.
    fn row(
        &mut self,
        flag: Flag,
        fields: Vec<String>,
        value: &dyn Rendered,
        default: Option<&dyn Rendered>,
    ) {
        self.row_shown(flag, fields, value, default, value.shown());
    }

    /// A row whose flag was typed otherwise than its fields read (a metric's
    /// spelling, an idle time the schema writes as for ever): `shown` is the
    /// flag's own value, for the refusal line; the fields keep theirs.
    fn row_shown(
        &mut self,
        flag: Flag,
        fields: Vec<String>,
        value: &dyn Rendered,
        default: Option<&dyn Rendered>,
        shown: String,
    ) {
        let rendered = value.value();
        debug_assert!(
            !rendered.is_null(),
            "a held field of {} renders null",
            flag.id
        );
        let (default, at_default) = match default {
            Some(default) => {
                let default = default.value();
                let at_default = default == rendered;
                (default, at_default)
            }
            None => (Value::Null, false),
        };
        if self.skips(&flag, at_default) {
            return;
        }
        let field_shown = value.shown();
        let fields = fields
            .into_iter()
            .map(|path| Field {
                path,
                value: rendered.clone(),
                shown: field_shown.clone(),
                presence: Presence::Held,
            })
            .collect();
        self.push(flag, fields, shown, rendered, default, at_default);
    }

    /// Consume a policy flag into two fields: the binding reference at
    /// `binding` (the profile's name, what the document holds there) and
    /// `profiles.<name>`, which holds the structure the flag fixes (see
    /// [`fixed_shape`]); the line shows the policy's name. A policy at its
    /// default is consumed only when the front end saw the flag, as a
    /// value is.
    /// The marks the profile builders left for `name`'s policy row, taken
    /// out of the implied set.
    fn claims(&mut self, name: &str) -> Vec<Field> {
        let prefix = format!("profiles.{name}.");
        let (mine, rest): (Vec<Field>, Vec<Field>) = self
            .implied
            .drain(..)
            .partition(|field| field.path.starts_with(&prefix));
        self.implied = rest;
        mine
    }

    fn profile_row(
        &mut self,
        flag: Flag,
        binding: &str,
        name: &str,
        profile: &ProfileSpec,
        policy: &PolicyConfig,
        default: Option<&str>,
    ) {
        let shown = policy.name();
        let at_default = default == Some(shown);
        // The profile's marks are the flag's claims: a row skipped (the flag
        // at its default, not given) makes none, so they go with it.
        let mine = self.claims(name);
        if self.skips(&flag, at_default) {
            return;
        }
        let mut fields = vec![
            Field {
                path: binding.to_string(),
                value: Value::String(name.to_string()),
                shown: name.to_string(),
                presence: Presence::Held,
            },
            Field {
                path: format!("profiles.{name}"),
                value: fixed_shape(policy, profile),
                shown: shown.to_string(),
                presence: Presence::Held,
            },
        ];
        fields.extend(mine);
        let rendered = Value::String(shown.to_string());
        let default = default.map_or(Value::Null, |default| Value::String(default.to_string()));
        self.push(
            flag,
            fields,
            shown.to_string(),
            rendered,
            default,
            at_default,
        );
    }

    /// A row for a flag that landed nowhere, with the reason; nothing when
    /// the flag has a row.
    fn unused_row(&mut self, flag: Flag, shown: String, reason: String) {
        if self.consumed.iter().any(|row| row.id == flag.id) {
            return;
        }
        self.consumed.push(Consumed {
            id: flag.id,
            flag: flag.flag,
            keyword: flag.keyword,
            fields: Vec::new(),
            shown,
            value: Value::Null,
            default: Value::Null,
            at_default: false,
            unused: Some(reason),
        });
    }

    /// One row per flag: a second consumption adds its fields, each with
    /// the value written there.
    fn push(
        &mut self,
        flag: Flag,
        fields: Vec<Field>,
        shown: String,
        value: Value,
        default: Value,
        at_default: bool,
    ) {
        if let Some(row) = self.consumed.iter_mut().find(|row| row.id == flag.id) {
            if row.unused.take().is_some() {
                row.shown = shown;
                row.value = value;
                row.default = default;
                row.at_default = at_default;
            }
            for field in fields {
                if !row.fields.iter().any(|known| known.path == field.path) {
                    row.fields.push(field);
                }
            }
            return;
        }
        self.consumed.push(Consumed {
            id: flag.id,
            flag: flag.flag,
            keyword: flag.keyword,
            fields,
            shown,
            value,
            default,
            at_default,
            unused: None,
        });
    }
}

/// The structure a policy flag fixes in its profile, as a document value:
/// the names of its score stages and its picker. The filters are the
/// document's. Where a knob flag of the policy governs the structure, the
/// flag fixes the knob's range: the power-of-two profile is one of its
/// metric's four forms (`forms`), the cache-aware picker one of the two
/// `--selection-temperature` selects; a knob flag given pins its own
/// choice on its row. The parameters are the knob rows' to check.
/// The power-of-two profile's four forms as a range, built once: the
/// shape of each metric's form, the request tokens not among them.
static POWER_OF_TWO_FORMS: LazyLock<Value> = LazyLock::new(|| {
    let forms: Vec<Value> = [
        PowerOfTwoLoadMetric::ExpectedWait,
        PowerOfTwoLoadMetric::Requests,
        PowerOfTwoLoadMetric::LeastRequests,
        PowerOfTwoLoadMetric::ResidentTokens { request_tokens: 0 },
    ]
    .into_iter()
    .map(|metric| {
        let (score, pick, _) = power_of_two_form(metric);
        serde_json::json!({ "score": [stage_name(&serde_json::to_value(&score).unwrap_or(Value::Null))], "pick": pick })
    })
    .collect();
    serde_json::json!({ "forms": forms })
});

fn fixed_shape(policy: &PolicyConfig, profile: &ProfileSpec) -> Value {
    if matches!(policy, PolicyConfig::PowerOfTwo { .. }) {
        return POWER_OF_TWO_FORMS.clone();
    }
    let mut shape = shape_of_value(&serde_json::to_value(profile).unwrap_or(Value::Null));
    if let Value::Object(map) = &mut shape {
        map.remove("filter");
        if matches!(policy, PolicyConfig::CacheAware { .. }) {
            map.insert(
                "pick".to_string(),
                serde_json::json!(["best", "best-with-temperature"]),
            );
        }
    }
    shape
}

/// Whether a document profile's shape holds what a policy flag fixes:
/// one of the `forms` when the flag fixes a range, else every stage name
/// the flag fixes is among the document's and the picker is the flag's
/// (one of the flag's, for a picker a knob selects) when the flag fixes
/// one. The flag fixes which stages, not their order: a document that
/// reorders them under the name passes.
fn shape_holds(found: &Value, fixed: &Value) -> bool {
    if let Some(Value::Array(forms)) = fixed.get("forms") {
        return forms.iter().any(|form| shape_holds(found, form));
    }
    let stages = |shape: &Value| {
        shape
            .get("score")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
    };
    let have = stages(found);
    let stages_hold = stages(fixed).iter().all(|stage| have.contains(stage));
    let pick_holds = match (fixed.get("pick"), found.get("pick")) {
        (None, _) => true,
        (Some(Value::Array(allowed)), Some(pick)) => allowed.contains(pick),
        (Some(pick), found) => found == Some(pick),
    };
    stages_hold && pick_holds
}

/// The name of a rendered stage or picker: a bare name is itself, one
/// written with parameters is its key.
fn stage_name(stage: &Value) -> Value {
    match stage {
        Value::Object(map) => map
            .keys()
            .next()
            .cloned()
            .map_or(Value::Null, Value::String),
        other => other.clone(),
    }
}

/// The shape of a rendered profile (see [`fixed_shape`]): a stage
/// written with parameters reduces to its name.
fn shape_of_value(profile: &Value) -> Value {
    let mut shape = serde_json::Map::new();
    if let Some(filter) = profile.get("filter") {
        shape.insert("filter".to_string(), filter.clone());
    }
    if let Some(Value::Array(stages)) = profile.get("score") {
        shape.insert(
            "score".to_string(),
            Value::Array(stages.iter().map(stage_name).collect()),
        );
    }
    if let Some(pick) = profile.get("pick") {
        shape.insert("pick".to_string(), stage_name(pick));
    }
    Value::Object(shape)
}

/// Whether a field path names a profile's score list whole
/// (`profiles.<name>.score`): such a field holds the stage names a flag pins,
/// present among the document's in any order; every other list compares
/// whole and in order.
fn is_score_path(path: &str) -> bool {
    let mut parts = path.split('.');
    parts.next() == Some("profiles")
        && parts.next().is_some()
        && parts.next() == Some("score")
        && parts.next().is_none()
}

/// Whether every stage name in `names` is among the document's `stages`.
fn stages_present(stages: &Value, names: &Value) -> bool {
    match (stages, names) {
        (Value::Array(stages), Value::Array(names)) => {
            let have: Vec<Value> = stages.iter().map(stage_name).collect();
            names.iter().all(|name| have.contains(name))
        }
        _ => false,
    }
}

/// Whether a field path names a profile whole (`profiles.<name>`): such a
/// field is compared by the profile's shape.
fn is_profile_path(path: &str) -> bool {
    let mut parts = path.split('.');
    parts.next() == Some("profiles") && parts.next().is_some() && parts.next().is_none()
}

/// A profile's shape as a line shows it: the parts it holds, in the
/// document's order; a range of forms as "one of" them.
fn render_shape(shape: &Value) -> String {
    if let Some(Value::Array(forms)) = shape.get("forms") {
        let forms: Vec<String> = forms.iter().map(render_shape).collect();
        return format!("one of: {}", forms.join("; "));
    }
    let parts: Vec<String> = ["filter", "score", "pick"]
        .iter()
        .filter_map(|key| {
            shape
                .get(*key)
                .map(|part| format!("{key} {}", render(part)))
        })
        .collect();
    if parts.is_empty() {
        "any structure".to_string()
    } else {
        parts.join(", ")
    }
}

impl Compiled {
    /// The conflict rule: every consumed flag against the
    /// `--policy-spec` document. `Ok` when the document holds each flag's
    /// value at its field (a field the document leaves out holds the
    /// default), the first flag whose value it does not hold otherwise.
    /// The policy names are checked first, so a document that names
    /// another profile is refused on `--policy` and not on a knob under the
    /// profile it does not have. A policy flag's profile field holds the
    /// structure the flag fixes, its stage names and its picker: a document
    /// may add filters and stages under the name, and chooses the parts a
    /// knob flag never given governs. A flag the mode leaves unused is not
    /// checked.
    pub fn check_against(&self, spec: &PolicySpec) -> Result<(), Disagreement> {
        let document = serde_json::to_value(spec).unwrap_or(Value::Null);
        let names = [
            POLICY.id,
            PREFILL_POLICY.id,
            DECODE_POLICY.id,
            ENCODE_POLICY.id,
        ];
        let is_name = |row: &&Consumed| names.contains(&row.id);
        let rows = self
            .report
            .consumed
            .iter()
            .filter(is_name)
            .chain(self.report.consumed.iter().filter(|row| !is_name(row)));
        for row in rows {
            for field in &row.fields {
                let found = lookup(&document, &field.path);
                let profile = is_profile_path(&field.path);
                let agrees = match (field.presence, found) {
                    (Presence::Present, found) => found.is_some(),
                    (Presence::Absent, found) => found.is_none(),
                    (Presence::Held, Some(found)) if profile => {
                        shape_holds(&shape_of_value(found), &field.value)
                    }
                    (Presence::Held, Some(found)) if is_score_path(&field.path) => {
                        stages_present(found, &field.value)
                    }
                    (Presence::Held, Some(found)) => holds(found, &field.value),
                    (Presence::Held, None) => !profile && same(&row.default, &field.value),
                };
                if !agrees {
                    let held = match found {
                        Some(found) if profile => render_shape(&shape_of_value(found)),
                        Some(found) => render(found),
                        None => "nothing".to_string(),
                    };
                    return Err(Disagreement {
                        flag: row.flag,
                        shown: row.shown.clone(),
                        field: field.path.clone(),
                        held,
                    });
                }
            }
        }
        Ok(())
    }
}

/// Whether a document value holds a flag's: a block's presence stands for
/// a switch flag's `true` and for the profile a policy flag names (the
/// row's `profiles.<name>` field), the values compare otherwise.
fn holds(found: &Value, value: &Value) -> bool {
    match (found, value) {
        (Value::Object(_), Value::Bool(switch)) => *switch,
        (Value::Object(_), Value::String(_)) => true,
        _ => same(found, value),
    }
}

/// Value equality with numbers compared as the document's single-precision
/// fields read them.
fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => match (x.as_f64(), y.as_f64()) {
            (Some(x), Some(y)) => (x - y).abs() <= 1e-6 * x.abs().max(1.0),
            _ => x == y,
        },
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(a, b)| same(a, b))
        }
        _ => a == b,
    }
}

/// One step of a field path.
enum Segment {
    Key(String),
    Index(usize),
}

/// The steps of a field path: keys by `.`, a list element by `[n]`, a map
/// key of any spelling by `["key"]`.
fn segments(path: &str) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut key = String::new();
    let mut chars = path.chars();
    while let Some(c) = chars.next() {
        match c {
            '.' => {
                if !key.is_empty() {
                    segments.push(Segment::Key(std::mem::take(&mut key)));
                }
            }
            '[' => {
                if !key.is_empty() {
                    segments.push(Segment::Key(std::mem::take(&mut key)));
                }
                let mut inner = String::new();
                for d in chars.by_ref() {
                    if d == ']' {
                        break;
                    }
                    inner.push(d);
                }
                let inner = inner.trim_matches('"');
                segments.push(match inner.parse::<usize>() {
                    Ok(index) => Segment::Index(index),
                    Err(_) => Segment::Key(inner.to_string()),
                });
            }
            _ => key.push(c),
        }
    }
    if !key.is_empty() {
        segments.push(Segment::Key(key));
    }
    segments
}

/// The value at `path` in a rendered document. A stage of a `score` list,
/// and a `pick`, are found by name; a bare name (no parameters written)
/// holds nothing further, so its parameters read as the defaults.
fn lookup<'a>(document: &'a Value, path: &str) -> Option<&'a Value> {
    let mut node = document;
    for segment in segments(path) {
        node = match (node, &segment) {
            (Value::Object(map), Segment::Key(key)) => map.get(key.as_str())?,
            (Value::Array(items), Segment::Index(index)) => items.get(*index)?,
            (Value::Array(items), Segment::Key(key)) => {
                items.iter().find_map(|item| match item {
                    Value::String(name) if name == key => Some(item),
                    Value::Object(map) => map.get(key.as_str()),
                    _ => None,
                })?
            }
            _ => return None,
        };
    }
    Some(node)
}

/// A document value as a line shows it: strings bare, single-precision
/// numbers short, lists in brackets.
fn render(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Number(number) => match number.as_f64() {
            Some(x) if x.fract() != 0.0 => {
                let narrow = x as f32;
                if f64::from(narrow) == x {
                    narrow.to_string()
                } else {
                    x.to_string()
                }
            }
            _ => number.to_string(),
        },
        Value::Array(items) => {
            let items: Vec<String> = items.iter().map(render).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Null => "null".to_string(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::allow_attributes
    )]

    use serde_json::json;
    use smg_policy_controller::SpecError;

    use super::*;
    use crate::config::{RouterConfig, RoutingKeyOverrideConfig};

    fn regular(policy: PolicyConfig) -> RouterConfig {
        RouterConfig::builder()
            .regular_mode(vec![
                "http://worker-a:8000".to_string(),
                "http://worker-b:8000".to_string(),
            ])
            .policy(policy)
            .build_unchecked()
    }

    fn cache_aware() -> PolicyConfig {
        PolicyConfig::cache_aware(0.3, 64, 1.5, 120, 67_108_864)
    }

    fn least_load() -> PolicyConfig {
        PolicyConfig::LeastLoad {
            kv_pressure_weight: 0.15,
            mean_prefill_tokens: 1024,
            default_throughput: 2000.0,
        }
    }

    fn doc(yaml: &str) -> PolicySpec {
        serde_yaml::from_str(yaml).expect("a well-formed document")
    }

    fn given(ids: &[&str]) -> GivenFlags {
        ids.iter().map(|id| (*id).to_string()).collect()
    }

    fn paths(row: &Consumed) -> Vec<&str> {
        row.fields.iter().map(|field| field.path.as_str()).collect()
    }

    fn ids(report: &Report) -> Vec<&'static str> {
        report.flags().collect()
    }

    /// The document every cache-aware command line at the defaults reads
    /// as.
    const DEFAULT_DOC: &str = "\
apiVersion: smg.policy/v1
generation: 1
profiles:
  cache_aware:
    filter: [healthy, not-overloaded, not-tried]
    score: [prefix-affinity, kv-pressure, least-load, warmup-slice]
    pick: best
bindings:
  - match: { model: \"*\" }
    profile: cache_aware
";

    /// A profile with no parameters, bound to every model.
    fn preset_doc(name: &str, score: &str, pick: &str) -> PolicySpec {
        doc(&format!(
            "apiVersion: smg.policy/v1\ngeneration: 1\nprofiles:\n  {name}:\n    filter: \
             [healthy, not-overloaded, not-tried]\n    score: {score}\n    pick: {pick}\n\
             bindings:\n  - match: {{ model: \"*\" }}\n    profile: {name}\n"
        ))
    }

    /// `--policy` left out: the default profile is written, the report
    /// says so and nothing is consumed.
    #[test]
    fn the_default_policy_is_written_and_noted_but_not_consumed() {
        let compiled = compile(&regular(cache_aware()));
        assert_eq!(compiled.spec, doc(DEFAULT_DOC));
        assert!(
            compiled.report.consumed.is_empty(),
            "{:?}",
            compiled.report.consumed
        );
        assert!(compiled.report.notes.contains(
            &"no --policy given: the default profile cache_aware was written".to_string()
        ));
    }

    /// A flag the front end saw is consumed at its default value too, with
    /// the deprecation line.
    #[test]
    fn a_given_flag_at_its_default_is_consumed_with_its_line() {
        let compiled = compile_given(
            &regular(cache_aware()),
            &given(&["policy", "cache_threshold"]),
        );
        assert_eq!(compiled.spec, doc(DEFAULT_DOC));
        assert_eq!(ids(&compiled.report), ["cache_threshold", "policy"]);
        assert!(compiled.report.consumed.iter().all(|row| row.at_default));
        let lines = compiled.report.deprecation_lines("1.17", false);
        assert_eq!(
            lines[0],
            "smg: --cache-threshold is deprecated and is removed in 1.17; it compiled to \
             profiles.cache_aware.score.prefix-affinity.threshold = 0.3; write it in \
             --policy-spec"
        );
        assert_eq!(
            lines[1],
            "smg: --policy is deprecated and is removed in 1.17; it compiled to \
             bindings[0].profile and profiles.cache_aware = cache_aware; write it in \
             --policy-spec"
        );
        let held = compiled.report.deprecation_lines("1.17", true);
        assert!(
            held[0].contains("= 0.3, which the spec already holds;"),
            "{}",
            held[0]
        );
    }

    /// Each parameterless preset expands to its profile.
    #[test]
    fn every_preset_expands_as_the_design_names_it() {
        let presets = [
            (least_load(), "least_load", "[least-load]", "best"),
            (
                PolicyConfig::PowerOfTwo {
                    load_metric: PowerOfTwoLoadMetric::ExpectedWait,
                },
                "power_of_two",
                "[least-load]",
                "power-of-two",
            ),
            (PolicyConfig::RoundRobin, "round_robin", "[]", "round-robin"),
            (PolicyConfig::Random, "random", "[]", "random"),
            (
                PolicyConfig::Passthrough,
                "passthrough",
                "[]",
                "passthrough",
            ),
        ];
        for (policy, name, score, pick) in presets {
            let compiled = compile(&regular(policy));
            assert_eq!(compiled.spec, preset_doc(name, score, pick), "{name}");
            assert_eq!(ids(&compiled.report), ["policy"], "{name}");
        }
    }

    /// The keyed presets: the ring lists its two headers in the view, the
    /// manual policy carries its sessions block, prefix hash its ring
    /// parameters and the boundaries.
    #[test]
    fn the_keyed_presets_carry_their_headers_blocks_and_parameters() {
        let ring = compile(&regular(PolicyConfig::ConsistentHashing));
        assert_eq!(
            ring.spec,
            doc("\
apiVersion: smg.policy/v1
generation: 1
view:
  headers: [x-smg-routing-key, x-smg-target-worker]
profiles:
  consistent_hashing:
    filter: [healthy, not-overloaded, not-tried]
    score: [consistent-hash]
    pick: best
bindings:
  - match: { model: \"*\" }
    profile: consistent_hashing
")
        );

        let manual = compile(&regular(PolicyConfig::Manual {
            eviction_interval_secs: 30,
            max_idle_secs: 3600,
            assignment_mode: ManualAssignmentMode::MinGroup,
        }));
        assert_eq!(
            manual.spec,
            doc("\
apiVersion: smg.policy/v1
generation: 1
view:
  headers: [x-smg-routing-key]
profiles:
  manual:
    filter: [healthy, not-overloaded]
    score: []
    pick: manual
bindings:
  - match: { model: \"*\" }
    profile: manual
    sessions:
      sticky_header: x-smg-routing-key
      headers: [x-smg-routing-key]
      ttl_s: 3600
      sweep_s: 30
      assignment: min_group
      inflight_cap: 2
      issue_session_id: false
")
        );
        assert_eq!(
            ids(&manual.report),
            [
                "max_idle_secs",
                "eviction_interval",
                "assignment_mode",
                "policy"
            ]
        );

        let mut prefix = regular(PolicyConfig::PrefixHash {
            prefix_token_count: 512,
            load_factor: 1.5,
            balance_abs_threshold: 4,
            cache_boundaries: vec![256, 1024, 4096],
        });
        prefix.cache_boundaries = vec![256, 1024, 4096];
        let compiled = compile(&prefix);
        assert_eq!(
            compiled.spec,
            doc("\
apiVersion: smg.policy/v1
generation: 1
view:
  boundaries: [256, 1024, 4096]
profiles:
  prefix_hash:
    filter: [healthy, not-overloaded, not-tried]
    score:
      - consistent-hash:
          prefix_tokens: 512
          bounded_load: { factor: 1.5, abs_margin: 4 }
    pick: best
bindings:
  - match: { model: \"*\" }
    profile: prefix_hash
")
        );
        assert_eq!(
            ids(&compiled.report),
            [
                "prefix_token_count",
                "prefix_hash_load_factor",
                "prefix_hash_balance_abs_threshold",
                "policy",
                "cache_boundaries"
            ]
        );
    }

    /// A parameter is written only off its default: the
    /// struct-era values of a programmatic caller land in the affinity
    /// stage and the per-model index map.
    #[test]
    fn a_parameter_is_written_only_off_its_default() {
        let compiled = compile(&regular(PolicyConfig::cache_aware(
            0.5, 32, 1.1, 60, 16_777_216,
        )));
        assert_eq!(
            compiled.spec,
            doc("\
apiVersion: smg.policy/v1
generation: 1
index:
  \"*\": { sweep_s: 60, max_size: 16777216 }
profiles:
  cache_aware:
    filter: [healthy, not-overloaded, not-tried]
    score:
      - prefix-affinity:
          threshold: 0.5
          spill: { abs: 32, rel: 1.1 }
      - kv-pressure
      - least-load
      - warmup-slice
    pick: best
bindings:
  - match: { model: \"*\" }
    profile: cache_aware
")
        );
        assert_eq!(
            ids(&compiled.report),
            [
                "cache_threshold",
                "balance_abs_threshold",
                "balance_rel_threshold",
                "max_tree_size",
                "eviction_interval"
            ]
        );
        let sweep = compiled
            .report
            .consumed
            .iter()
            .find(|row| row.id == "eviction_interval")
            .expect("the sweep row");
        assert_eq!(
            sweep.keyword_warning("1.17"),
            "smg: the keyword eviction_interval_secs is deprecated and is removed in 1.17; it \
             compiled to index[\"*\"].sweep_s = 60; write it in policy_spec"
        );
    }

    /// The hash index and every event-driven knob land in their stages,
    /// the temperature in the picker, the boundaries in the view.
    #[test]
    fn the_hash_index_and_every_event_knob_land_in_their_stages() {
        let mut config = RouterConfig::builder()
            .regular_mode(vec!["http://worker-a:8000".to_string()])
            .policy(PolicyConfig::CacheAware {
                cache_threshold: 0.3,
                balance_abs_threshold: 64,
                balance_rel_threshold: 1.5,
                balance_prefix_margin_tokens: 1024,
                balance_uncached_credit: true,
                eviction_interval_secs: 120,
                max_tree_size: 67_108_864,
                block_size: 16,
                balance_token_usage_threshold: 0.3,
                overload_token_usage_threshold: 0.9,
                overlap_decay: 0.5,
                selection_temperature: 0.05,
                cache_index: CacheIndexKind::Hash,
                cache_ttl_secs: 300,
                cache_boundaries: vec![512, 2048],
                selection_policy: None,
                selection_accounting_ttl_ms: 200,
            })
            .worker_warmup(0, 0.25, 1024, 0.5, 8)
            .build_unchecked();
        config.cache_boundaries = vec![512, 2048];
        let compiled = compile(&config);
        assert_eq!(
            compiled.spec,
            doc("\
apiVersion: smg.policy/v1
generation: 1
view:
  boundaries: [512, 2048]
index:
  \"*\": { kind: hash, ttl_s: 300 }
profiles:
  cache_aware:
    filter: [healthy, not-overloaded, not-tried]
    score:
      - prefix-affinity:
          decay: 0.5
          accounting_ttl_ms: 200
          spill: { prefix_margin_tokens: 1024, uncached_credit: true }
      - kv-pressure: { spread: 0.3, ceiling: 0.9 }
      - least-load
      - warmup-slice: { secs: 0 }
    pick: { best-with-temperature: { temperature: 0.05 } }
bindings:
  - match: { model: \"*\" }
    profile: cache_aware
")
        );
        assert_eq!(
            ids(&compiled.report),
            [
                "overlap_decay",
                "selection_accounting_ttl_ms",
                "balance_prefix_margin_tokens",
                "balance_uncached_credit",
                "balance_token_usage_threshold",
                "overload_token_usage_threshold",
                "worker_warmup_secs",
                "selection_temperature",
                "cache_index",
                "cache_ttl_secs",
                "cache_boundaries"
            ]
        );
    }

    /// Least load carries its tunable, the sticky override its block on
    /// custom headers, and the sweep stays written at the command line's
    /// value.
    #[test]
    fn least_load_carries_its_tunable_and_the_sticky_block_on_custom_headers() {
        let config = RouterConfig::builder()
            .regular_mode(vec!["http://worker-a:8000".to_string()])
            .policy(PolicyConfig::LeastLoad {
                kv_pressure_weight: 0.3,
                mean_prefill_tokens: 1024,
                default_throughput: 2000.0,
            })
            .routing_key_override(RoutingKeyOverrideConfig {
                enabled: true,
                eviction_interval_secs: 120,
                max_idle_secs: 600,
                assignment_mode: ManualAssignmentMode::MinLoad,
                headers: vec!["x-session-id".to_string(), "x-smg-routing-key".to_string()],
            })
            .build_unchecked();
        let compiled = compile(&config);
        assert_eq!(
            compiled.spec,
            doc("\
apiVersion: smg.policy/v1
generation: 1
view:
  headers: [x-session-id, x-smg-routing-key]
profiles:
  least_load:
    filter: [healthy, not-overloaded, not-tried]
    score:
      - least-load: { kv_pressure_weight: 0.3 }
    pick: best
bindings:
  - match: { model: \"*\" }
    profile: least_load
    sessions:
      sticky_header: x-session-id
      headers: [x-session-id, x-smg-routing-key]
      ttl_s: 600
      sweep_s: 120
      assignment: min_load
      inflight_cap: 2
      issue_session_id: false
")
        );
        assert_eq!(
            ids(&compiled.report),
            [
                "least_load_kv_pressure_weight",
                "policy",
                "routing_key_override",
                "routing_key_headers",
                "max_idle_secs",
                "assignment_mode"
            ]
        );
    }

    /// `--eviction-interval` lands in the index map and the sticky block
    /// at once, as one row with two fields; the delegated override keeps
    /// the latest pin.
    #[test]
    fn the_eviction_interval_lands_in_the_index_and_the_sticky_block_at_once() {
        let config = RouterConfig::builder()
            .regular_mode(vec!["http://worker-a:8000".to_string()])
            .policy(PolicyConfig::cache_aware(0.4, 48, 1.3, 90, 67_108_864))
            .routing_key_override(RoutingKeyOverrideConfig {
                enabled: true,
                eviction_interval_secs: 90,
                max_idle_secs: 600,
                ..RoutingKeyOverrideConfig::default()
            })
            .build_unchecked();
        let compiled = compile(&config);
        assert_eq!(
            compiled.spec,
            doc("\
apiVersion: smg.policy/v1
generation: 1
view:
  headers: [x-smg-routing-key]
index:
  \"*\": { sweep_s: 90 }
profiles:
  cache_aware:
    filter: [healthy, not-overloaded, not-tried]
    score:
      - prefix-affinity:
          threshold: 0.4
          spill: { abs: 48, rel: 1.3 }
      - kv-pressure
      - least-load
      - warmup-slice
    pick: best
bindings:
  - match: { model: \"*\" }
    profile: cache_aware
    sessions:
      sticky_header: x-smg-routing-key
      headers: [x-smg-routing-key]
      ttl_s: 600
      sweep_s: 90
      assignment: delegate
      inflight_cap: 2
      issue_session_id: false
      recovery: latest
")
        );
        let sweep = compiled
            .report
            .consumed
            .iter()
            .find(|row| row.id == "eviction_interval")
            .expect("the sweep row");
        assert_eq!(
            paths(sweep),
            ["index[\"*\"].sweep_s", "bindings[0].sessions.sweep_s"]
        );
        assert_eq!(sweep.shown, "90");
    }

    fn pair(prefill: Option<PolicyConfig>, decode: Option<PolicyConfig>) -> RouterConfig {
        RouterConfig::builder()
            .mode(RoutingMode::PrefillDecode {
                prefill_urls: vec![("http://prefill-a:8000".to_string(), Some(8998))],
                decode_urls: vec!["http://decode-a:8000".to_string()],
                prefill_policy: prefill,
                decode_policy: decode,
            })
            .policy(cache_aware())
            .build_unchecked()
    }

    /// Both sides named: each side's profile with its capacity filter
    /// under the `pd-pair` plan, the main policy unused and noted; the main
    /// policy as both sides' fallback with strict pairing.
    #[test]
    fn the_pair_plan_names_both_sides_with_their_capacity_filters() {
        let both = compile_given(
            &pair(
                Some(PolicyConfig::RoundRobin),
                Some(PolicyConfig::RoundRobin),
            ),
            &given(&["policy"]),
        );
        assert_eq!(
            both.spec,
            doc("\
apiVersion: smg.policy/v1
generation: 1
profiles:
  prefill:
    filter: [healthy, not-overloaded, has-capacity, not-tried]
    score: []
    pick: round-robin
  decode:
    filter: [healthy, not-overloaded, has-room, not-tried]
    score: []
    pick: round-robin
bindings:
  - match: { model: \"*\", phase: [prefill, decode] }
    plan: pd-pair
    profiles: { prefill: prefill, decode: decode }
")
        );
        assert_eq!(
            ids(&both.report),
            ["prefill_policy", "decode_policy", "policy"]
        );
        assert_eq!(
            both.report.consumed[0].deprecation_line("1.17", false),
            "smg: --prefill-policy is deprecated and is removed in 1.17; it compiled to \
             bindings[0].profiles.prefill = prefill and profiles.prefill = round_robin; write it \
             in --policy-spec"
        );
        let unused = both.report.deprecation_lines("1.17", false)[2].clone();
        assert!(
            unused.contains(
                "it compiled to nothing (cache_aware routes no request under --pd-disaggregation)"
            ),
            "{unused}"
        );
        assert!(both.report.notes.iter().any(|note| note
            == "the main policy cache_aware routes no request under --pd-disaggregation: it compiled to nothing"));

        let mut fallback = pair(None, None);
        fallback.pd_pairing_mode = PdPairingMode::Strict;
        let compiled = compile_given(&fallback, &given(&["policy"]));
        assert_eq!(
            compiled.spec,
            doc("\
apiVersion: smg.policy/v1
generation: 1
profiles:
  prefill:
    filter: [healthy, not-overloaded, has-capacity, not-tried]
    score: [prefix-affinity, kv-pressure, least-load, warmup-slice]
    pick: best
  decode:
    filter: [healthy, not-overloaded, has-room, not-tried]
    score: [prefix-affinity, kv-pressure, least-load, warmup-slice]
    pick: best
plans:
  pd-pair: { pairing: strict }
bindings:
  - match: { model: \"*\", phase: [prefill, decode] }
    plan: pd-pair
    profiles: { prefill: prefill, decode: decode }
")
        );
        assert_eq!(ids(&compiled.report), ["policy", "pd_pairing_mode"]);
        let main = &compiled.report.consumed[0];
        assert_eq!(
            paths(main),
            [
                "bindings[0].profiles.prefill",
                "profiles.prefill",
                "bindings[0].profiles.decode",
                "profiles.decode"
            ]
        );
    }

    /// The encode side joins under the `epd` plan; its default is written
    /// and noted when no encode policy is given.
    #[test]
    fn the_epd_plan_adds_the_encode_side_and_names_its_default() {
        let epd = |encode: Option<PolicyConfig>| {
            RouterConfig::builder()
                .mode(RoutingMode::EncodePrefillDecode {
                    encode_urls: vec![("http://encode-a:8000".to_string(), Some(8998))],
                    prefill_urls: vec![("http://prefill-a:8000".to_string(), Some(8998))],
                    decode_urls: vec!["http://decode-a:8000".to_string()],
                    encode_policy: encode,
                    prefill_policy: None,
                    decode_policy: None,
                })
                .policy(cache_aware())
                .build_unchecked()
        };
        let expected = doc("\
apiVersion: smg.policy/v1
generation: 1
profiles:
  encode:
    filter: [healthy, not-overloaded, not-tried]
    score: [consistent-hash]
    pick: best
  prefill:
    filter: [healthy, not-overloaded, has-capacity, not-tried]
    score: [prefix-affinity, kv-pressure, least-load, warmup-slice]
    pick: best
  decode:
    filter: [healthy, not-overloaded, has-room, not-tried]
    score: [prefix-affinity, kv-pressure, least-load, warmup-slice]
    pick: best
bindings:
  - match: { model: \"*\", phase: [encode, prefill, decode] }
    plan: epd
    profiles: { encode: encode, prefill: prefill, decode: decode }
");
        let named = compile(&epd(Some(PolicyConfig::ConsistentHashing)));
        assert_eq!(named.spec, expected);
        assert_eq!(ids(&named.report), ["encode_policy"]);
        assert!(
            named.spec.view.headers.is_empty(),
            "the encode ring keys by item"
        );

        let defaulted = compile(&epd(None));
        assert_eq!(defaulted.spec, expected);
        assert!(defaulted.report.consumed.is_empty());
        assert!(defaulted.report.notes.iter().any(|note| {
            note
            == "no --encode-policy given: the default encode profile consistent_hashing was written"
        }));
    }

    /// A bucket prefill keeps its two filters before its capacity filter,
    /// beside a power-of-two decode.
    #[test]
    fn a_bucket_prefill_pairs_with_a_power_of_two_decode() {
        let compiled = compile(&pair(
            Some(PolicyConfig::Bucket {
                balance_abs_threshold: 64,
                balance_rel_threshold: 1.5,
                bucket_adjust_interval_secs: 5,
            }),
            Some(PolicyConfig::PowerOfTwo {
                load_metric: PowerOfTwoLoadMetric::ExpectedWait,
            }),
        ));
        assert_eq!(
            compiled.spec,
            doc("\
apiVersion: smg.policy/v1
generation: 1
profiles:
  prefill:
    filter: [healthy, not-overloaded, has-capacity]
    score: [bucket]
    pick: best
  decode:
    filter: [healthy, not-overloaded, has-room, not-tried]
    score: [least-load]
    pick: power-of-two
bindings:
  - match: { model: \"*\", phase: [prefill, decode] }
    plan: pd-pair
    profiles: { prefill: prefill, decode: decode }
")
        );
    }

    /// The overload thresholds and the shed rule compile to the binding's
    /// block; protection off compiles to `enabled: false`
    /// under the negative flag's name.
    #[test]
    fn the_overload_flags_compile_to_the_binding_block() {
        let tightened = compile(
            &RouterConfig::builder()
                .regular_mode(vec!["http://worker-a:8000".to_string()])
                .policy(cache_aware())
                .worker_overload_waiting_requests(Some(32))
                .worker_overload_token_usage(Some(0.9))
                .worker_overload_shed(true)
                .build_unchecked(),
        );
        assert_eq!(
            tightened.spec.bindings[0].overload,
            Some(OverloadSpec {
                enabled: true,
                waiting_requests: Some(32),
                token_usage: Some(0.9),
                shed_when_all_vetoed: true,
            })
        );
        assert_eq!(
            ids(&tightened.report),
            [
                "worker_overload_waiting_requests",
                "worker_overload_token_usage",
                "worker_overload_shed"
            ]
        );
        assert!(tightened
            .report
            .notes
            .contains(&PER_WORKER_OVERLOAD_NOTE.to_string()));

        let off = compile(
            &RouterConfig::builder()
                .regular_mode(vec!["http://worker-a:8000".to_string()])
                .policy(PolicyConfig::RoundRobin)
                .worker_overload_protection(false)
                .build_unchecked(),
        );
        assert_eq!(
            off.spec.bindings[0].overload,
            Some(OverloadSpec {
                enabled: false,
                ..OverloadSpec::default()
            })
        );
        assert_eq!(
            ids(&off.report),
            ["policy", "disable_worker_overload_protection"]
        );
        assert_eq!(
            off.report.deprecation_lines("1.17", false)[1],
            "smg: --disable-worker-overload-protection is deprecated and is removed in 1.17; it \
             compiled to bindings[0].overload.enabled = false; write it in --policy-spec"
        );
    }

    /// The request-count forms of `power_of_two` compare the requests in
    /// flight through the engine's count-first pickers; the resident-tokens
    /// form scores every candidate's resident work.
    #[test]
    fn power_of_two_forms_follow_the_engines_count_first_pickers() {
        let form = |metric| {
            compile(&regular(PolicyConfig::PowerOfTwo {
                load_metric: metric,
            }))
        };
        let resident = form(PowerOfTwoLoadMetric::ResidentTokens {
            request_tokens: 512,
        });
        assert_eq!(
            resident.spec,
            doc("\
apiVersion: smg.policy/v1
generation: 1
profiles:
  power_of_two:
    filter: [healthy, not-overloaded, not-tried]
    score:
      - resident-work: { request_tokens: 512 }
    pick: best
bindings:
  - match: { model: \"*\" }
    profile: power_of_two
")
        );
        assert_eq!(
            ids(&resident.report),
            ["power_of_two_load_metric", "policy"]
        );
        let requests = form(PowerOfTwoLoadMetric::Requests);
        assert_eq!(
            requests.spec,
            preset_doc("power_of_two", "[least-load]", "power-of-two-count-first")
        );
        let fewest = form(PowerOfTwoLoadMetric::LeastRequests);
        assert_eq!(
            fewest.spec,
            preset_doc("power_of_two", "[least-load]", "count-first")
        );
    }

    /// A flag with no home in the document leaves no row.
    #[test]
    fn kept_flags_leave_no_row() {
        let compiled = compile(
            &RouterConfig::builder()
                .regular_mode(vec!["http://engine-a:8000".to_string()])
                .policy(least_load())
                .dp_aware(true)
                .dp_minimum_tokens_scheduler(true)
                .build_unchecked(),
        );
        assert_eq!(ids(&compiled.report), ["policy"]);
        assert_eq!(
            compiled.spec,
            preset_doc("least_load", "[least-load]", "best")
        );
    }

    /// A flag whose value the document does not hold is refused naming
    /// both; flags the document holds pass; a field the document leaves out
    /// holds the default.
    #[test]
    fn a_flag_that_disagrees_with_the_document_is_refused_and_agreeing_ones_pass() {
        let document = doc(DEFAULT_DOC);
        let least_load = compile_given(&regular(least_load()), &given(&["policy"]));
        let refused = least_load
            .check_against(&document)
            .expect_err("another profile is refused");
        assert_eq!(
            refused.to_string(),
            "--policy least_load disagrees with --policy-spec bindings[0].profile = \
             cache_aware; remove the flag or change the spec (the spec is the source of truth)"
        );

        let agreeing = compile_given(
            &regular(cache_aware()),
            &given(&["policy", "cache_threshold"]),
        );
        assert_eq!(agreeing.check_against(&document), Ok(()));

        let tuned = doc("\
apiVersion: smg.policy/v1
generation: 1
profiles:
  cache_aware:
    filter: [healthy, not-overloaded, not-tried]
    score:
      - prefix-affinity: { threshold: 0.5 }
      - kv-pressure
      - least-load
      - warmup-slice
    pick: best
bindings:
  - match: { model: \"*\" }
    profile: cache_aware
");
        let refused = agreeing
            .check_against(&tuned)
            .expect_err("a knob the document holds otherwise is refused");
        assert_eq!(refused.flag, "--cache-threshold");
        assert_eq!(
            refused.field,
            "profiles.cache_aware.score.prefix-affinity.threshold"
        );
        assert_eq!(refused.held, "0.5");

        let off_default = compile(&regular(PolicyConfig::cache_aware(
            0.5, 64, 1.5, 120, 67_108_864,
        )));
        assert_eq!(off_default.check_against(&tuned), Ok(()));
        let refused = off_default
            .check_against(&document)
            .expect_err("a bare stage holds the default");
        assert_eq!(refused.held, "nothing");
    }

    /// The path lookup: a stage by its name in the score list, a bare
    /// stage holding nothing further, a list index and a quoted map key.
    #[test]
    fn the_lookup_reads_stages_by_name_and_a_bare_stage_holds_nothing_further() {
        let document = json!({
            "profiles": { "p": {
                "score": ["least-load", { "prefix-affinity": { "threshold": 0.5 } }],
                "pick": "best"
            } },
            "bindings": [{ "profile": "p" }],
            "index": { "*": { "sweep_s": 60 } }
        });
        assert_eq!(
            lookup(&document, "profiles.p.score.prefix-affinity.threshold"),
            Some(&json!(0.5))
        );
        assert_eq!(
            lookup(&document, "profiles.p.score.least-load.kv_pressure_weight"),
            None
        );
        assert_eq!(
            lookup(
                &document,
                "profiles.p.pick.best-with-temperature.temperature"
            ),
            None
        );
        assert_eq!(lookup(&document, "bindings[0].profile"), Some(&json!("p")));
        assert_eq!(lookup(&document, "index[\"*\"].sweep_s"), Some(&json!(60)));
        assert_eq!(lookup(&document, "bindings[1].profile"), None);
    }

    /// A zero of either legacy setting keeps keys for ever: both schema
    /// fields read as their largest value.
    #[test]
    fn a_zero_idle_or_sweep_keeps_keys_for_ever() {
        assert_eq!(lifetime(0, 60), (u32::MAX, u32::MAX));
        assert_eq!(lifetime(600, 0), (u32::MAX, u32::MAX));
        assert_eq!(lifetime(600, 60), (600, 60));
    }

    /// The removal release is one reviewed constant in the form of a minor
    /// release: the one after the first release reading `--policy-spec`.
    #[test]
    fn the_removal_release_is_the_minor_after_the_reader() {
        let (major, minor) = ROUTING_FLAGS_REMOVED_IN
            .split_once('.')
            .expect("a major.minor release");
        assert!(major.parse::<u32>().is_ok(), "{ROUTING_FLAGS_REMOVED_IN}");
        assert!(minor.parse::<u32>().is_ok(), "{ROUTING_FLAGS_REMOVED_IN}");
        assert_eq!(ROUTING_FLAGS_REMOVED_IN, "1.17");
    }

    /// The prefix form is written whole: the engine reads the prefix from
    /// the presence of `prefix_tokens` and the bound from its block, so a
    /// default `prefix_hash` carries both.
    #[test]
    fn a_default_prefix_hash_is_written_whole() {
        let compiled = compile(&regular(PolicyConfig::PrefixHash {
            prefix_token_count: 256,
            load_factor: 1.25,
            balance_abs_threshold: 10,
            cache_boundaries: Vec::new(),
        }));
        assert_eq!(
            compiled.spec,
            doc("\
apiVersion: smg.policy/v1
generation: 1
profiles:
  prefix_hash:
    filter: [healthy, not-overloaded, not-tried]
    score:
      - consistent-hash:
          prefix_tokens: 256
          bounded_load: { factor: 1.25, abs_margin: 10 }
    pick: best
bindings:
  - match: { model: \"*\" }
    profile: prefix_hash
")
        );
        assert_eq!(ids(&compiled.report), ["policy"]);
    }

    /// A routing flag the front end saw that the policy in use does not
    /// read still gets its row: the line names the reason instead of a
    /// field, and the series counts it.
    #[test]
    fn a_given_flag_the_policy_does_not_read_gets_a_row_naming_the_reason() {
        let compiled = compile_given(
            &regular(PolicyConfig::RoundRobin),
            &given(&["policy", "cache_threshold", "max_idle_secs"]),
        );
        assert_eq!(
            ids(&compiled.report),
            ["policy", "cache_threshold", "max_idle_secs"]
        );
        let unread = &compiled.report.consumed[1];
        assert!(unread.fields.is_empty());
        assert_eq!(
            unread.deprecation_line("1.17", false),
            "smg: --cache-threshold is deprecated and is removed in 1.17; it compiled to \
             nothing (the round_robin policy does not read it); write it in --policy-spec"
        );
        assert_eq!(compiled.check_against(&compiled.spec), Ok(()));
    }

    /// A zero idle time keeps keys for ever and owns the sweep field too; no
    /// row claims a sweep that was never given.
    #[test]
    fn a_zero_idle_time_owns_the_sweep_field() {
        let compiled = compile(&regular(PolicyConfig::Manual {
            eviction_interval_secs: 120,
            max_idle_secs: 0,
            assignment_mode: ManualAssignmentMode::Random,
        }));
        assert_eq!(ids(&compiled.report), ["max_idle_secs", "policy"]);
        let idle = &compiled.report.consumed[0];
        assert_eq!(
            paths(idle),
            ["bindings[0].sessions.ttl_s", "bindings[0].sessions.sweep_s"]
        );
        assert_eq!(
            idle.deprecation_line("1.17", false),
            "smg: --max-idle-secs is deprecated and is removed in 1.17; it compiled to \
             bindings[0].sessions.ttl_s and bindings[0].sessions.sweep_s = 4294967295; write \
             it in --policy-spec"
        );
        assert_eq!(compiled.check_against(&compiled.spec), Ok(()));

        let carried = compile_given(
            &regular(PolicyConfig::Manual {
                eviction_interval_secs: 60,
                max_idle_secs: 0,
                assignment_mode: ManualAssignmentMode::Random,
            }),
            &given(&["max_idle_secs", "eviction_interval"]),
        );
        assert_eq!(
            ids(&carried.report),
            ["max_idle_secs", "eviction_interval", "policy"]
        );
        assert_eq!(
            carried.report.consumed[1].deprecation_line("1.17", false),
            "smg: --eviction-interval is deprecated and is removed in 1.17; it compiled to \
             nothing (a zero idle time keeps keys for ever, so the sweep is for ever too); \
             write it in --policy-spec"
        );
        assert_eq!(carried.check_against(&carried.spec), Ok(()));
    }

    /// The resident-tokens form names its request tokens beside the picker.
    #[test]
    fn the_resident_form_names_its_request_tokens() {
        let compiled = compile(&regular(PolicyConfig::PowerOfTwo {
            load_metric: PowerOfTwoLoadMetric::ResidentTokens {
                request_tokens: 4096,
            },
        }));
        let row = &compiled.report.consumed[0];
        assert_eq!(row.id, "power_of_two_load_metric");
        assert_eq!(
            paths(row),
            [
                "profiles.power_of_two.pick",
                "profiles.power_of_two.score",
                "profiles.power_of_two.score.resident-work.request_tokens"
            ]
        );
        assert_eq!(
            row.deprecation_line("1.17", false),
            "smg: --power-of-two-load-metric is deprecated and is removed in 1.17; it compiled \
             to profiles.power_of_two.pick = best and profiles.power_of_two.score = \
             [resident-work] and profiles.power_of_two.score.resident-work.request_tokens = \
             4096; write it in \
             --policy-spec"
        );
        assert_eq!(compiled.check_against(&compiled.spec), Ok(()));
    }

    /// A document that binds another profile under a side is refused on
    /// the side's flag, the binding field named.
    #[test]
    fn a_document_binding_another_profile_under_the_side_is_refused() {
        let compiled = compile(&pair(
            Some(PolicyConfig::RoundRobin),
            Some(PolicyConfig::RoundRobin),
        ));
        let mut document = serde_json::to_value(&compiled.spec).unwrap();
        document["bindings"][0]["profiles"]["prefill"] = Value::String("decode".to_string());
        let rebound: PolicySpec = serde_json::from_value(document).unwrap();
        let refused = compiled
            .check_against(&rebound)
            .expect_err("the prefill side bound to the decode profile");
        assert_eq!(refused.flag, "--prefill-policy");
        assert_eq!(refused.field, "bindings[0].profiles.prefill");
        assert_eq!(refused.held, "decode");
        assert_eq!(compiled.check_against(&compiled.spec), Ok(()));
    }

    /// A flag one side leaves unused and the other consumes keeps the
    /// consumption's value and default, so the pair's own document passes.
    #[test]
    fn a_flag_unused_by_one_side_and_consumed_by_the_other_round_trips() {
        let compiled = compile_given(
            &pair(
                Some(PolicyConfig::Manual {
                    eviction_interval_secs: 120,
                    max_idle_secs: 0,
                    assignment_mode: ManualAssignmentMode::Random,
                }),
                Some(cache_aware()),
            ),
            &given(&["max_idle_secs", "eviction_interval"]),
        );
        let sweep = compiled
            .report
            .consumed
            .iter()
            .find(|row| row.id == "eviction_interval")
            .expect("the sweep row");
        assert_eq!(paths(sweep), ["index[\"*\"].sweep_s"]);
        assert_eq!(sweep.default, serde_json::json!(120));
        assert!(sweep.at_default && sweep.unused.is_none());
        assert_eq!(compiled.check_against(&compiled.spec), Ok(()));
    }

    /// A policy flag fixes its profile's stage names and picker, not its
    /// filters and not what a knob flag never given governs: a document may
    /// add a filter or a stage under the name, and the power-of-two profile
    /// follows the metric knob whole.
    #[test]
    fn a_knob_the_flag_never_set_is_the_documents_choice() {
        let compiled = compile_given(&regular(cache_aware()), &given(&["policy"]));
        let mut document = serde_json::to_value(&compiled.spec).unwrap();
        let profile = &mut document["profiles"]["cache_aware"];
        profile["filter"]
            .as_array_mut()
            .unwrap()
            .push(Value::String("has-capacity".to_string()));
        profile["score"]
            .as_array_mut()
            .unwrap()
            .push(Value::String("least-load".to_string()));
        let widened: PolicySpec = serde_json::from_value(document.clone()).unwrap();
        assert_eq!(compiled.check_against(&widened), Ok(()));
        document["profiles"]["cache_aware"]["pick"] =
            serde_json::json!({ "best-with-temperature": { "temperature": 0.5 } });
        let tempered: PolicySpec = serde_json::from_value(document.clone()).unwrap();
        assert_eq!(compiled.check_against(&tempered), Ok(()));
        document["profiles"]["cache_aware"]["score"] = serde_json::json!(["least-load"]);
        let narrowed: PolicySpec = serde_json::from_value(document).unwrap();
        assert_eq!(
            compiled
                .check_against(&narrowed)
                .expect_err("a stage removed")
                .field,
            "profiles.cache_aware"
        );

        let two = compile_given(
            &regular(PolicyConfig::PowerOfTwo {
                load_metric: PowerOfTwoLoadMetric::ExpectedWait,
            }),
            &given(&["policy"]),
        );
        let requests = doc("\
apiVersion: smg.policy/v1
generation: 1
profiles:
  power_of_two:
    filter: [healthy, not-overloaded, not-tried]
    score: [least-load]
    pick: power-of-two-count-first
bindings:
  - match: { model: \"*\" }
    profile: power_of_two
");
        assert_eq!(two.check_against(&requests), Ok(()));
    }

    /// A knob flag given pins the whole structure it governs: under a given
    /// metric the power-of-two scorer is the metric's, a picker outside the
    /// temperature knob's two is refused under cache_aware, and a picker
    /// outside the four metric forms is refused under power_of_two.
    #[test]
    fn a_knob_given_pins_its_structure_and_a_range_bounds_the_rest() {
        let two = |metric| {
            regular(PolicyConfig::PowerOfTwo {
                load_metric: metric,
            })
        };
        let given_metric = compile_given(
            &two(PowerOfTwoLoadMetric::ExpectedWait),
            &given(&["policy", "power_of_two_load_metric"]),
        );
        let mut document = serde_json::to_value(&given_metric.spec).unwrap();
        document["profiles"]["power_of_two"]["score"] = serde_json::json!(["resident-work"]);
        document["profiles"]["power_of_two"]["pick"] = serde_json::json!("best");
        let swapped: PolicySpec = serde_json::from_value(document).unwrap();
        let refused = given_metric
            .check_against(&swapped)
            .expect_err("the form swapped under a given metric");
        assert_eq!(refused.flag, "--power-of-two-load-metric");
        assert_eq!(refused.field, "profiles.power_of_two.pick");
        let metric = given_metric
            .report
            .consumed
            .iter()
            .find(|row| row.id == "power_of_two_load_metric")
            .unwrap();
        assert_eq!(
            paths(metric),
            ["profiles.power_of_two.pick", "profiles.power_of_two.score"]
        );
        assert_eq!(given_metric.check_against(&given_metric.spec), Ok(()));

        let bare = compile_given(
            &two(PowerOfTwoLoadMetric::ExpectedWait),
            &given(&["policy"]),
        );
        let mut document = serde_json::to_value(&bare.spec).unwrap();
        document["profiles"]["power_of_two"]["pick"] = Value::String("round-robin".to_string());
        let foreign: PolicySpec = serde_json::from_value(document).unwrap();
        assert_eq!(
            bare.check_against(&foreign)
                .expect_err("a picker outside the four forms")
                .field,
            "profiles.power_of_two"
        );

        let aware = compile_given(&regular(cache_aware()), &given(&["policy"]));
        let mut document = serde_json::to_value(&aware.spec).unwrap();
        document["profiles"]["cache_aware"]["pick"] = Value::String("random".to_string());
        let random: PolicySpec = serde_json::from_value(document).unwrap();
        assert_eq!(
            aware
                .check_against(&random)
                .expect_err("a picker outside the knob's two")
                .field,
            "profiles.cache_aware"
        );
    }

    /// The prefix form and its bound are read from the presence of their
    /// fields, so `--policy prefix_hash` implies them: a document with plain
    /// hashing under the name is refused on the policy flag.
    #[test]
    fn a_plain_hash_document_is_refused_under_the_prefix_hash_flag() {
        let compiled = compile_given(
            &regular(PolicyConfig::PrefixHash {
                prefix_token_count: 256,
                load_factor: 1.25,
                balance_abs_threshold: 10,
                cache_boundaries: Vec::new(),
            }),
            &given(&["policy"]),
        );
        assert_eq!(ids(&compiled.report), ["policy"]);
        assert_eq!(
            paths(&compiled.report.consumed[0]),
            [
                "bindings[0].profile",
                "profiles.prefix_hash",
                "profiles.prefix_hash.score.consistent-hash.prefix_tokens",
                "profiles.prefix_hash.score.consistent-hash.bounded_load"
            ]
        );
        let plain = doc("\
apiVersion: smg.policy/v1
generation: 1
profiles:
  prefix_hash:
    filter: [healthy, not-overloaded, not-tried]
    score: [consistent-hash]
    pick: best
bindings:
  - match: { model: \"*\" }
    profile: prefix_hash
");
        let refused = compiled
            .check_against(&plain)
            .expect_err("plain hashing under the name");
        assert_eq!(refused.flag, "--policy");
        assert_eq!(
            refused.field,
            "profiles.prefix_hash.score.consistent-hash.prefix_tokens"
        );
        assert_eq!(compiled.check_against(&compiled.spec), Ok(()));
        let mut longer = serde_json::to_value(&compiled.spec).unwrap();
        longer["profiles"]["prefix_hash"]["score"][0]["consistent-hash"]["prefix_tokens"] =
            serde_json::json!(512);
        let longer: PolicySpec = serde_json::from_value(longer).unwrap();
        assert_eq!(compiled.check_against(&longer), Ok(()));
    }

    /// A front end without a given set still gets the lifetime unused rows
    /// when the raw value is off its default.
    #[test]
    fn an_idle_time_off_default_beside_a_zero_sweep_has_its_row_without_a_given_set() {
        let compiled = compile(&regular(PolicyConfig::Manual {
            eviction_interval_secs: 0,
            max_idle_secs: 600,
            assignment_mode: ManualAssignmentMode::Random,
        }));
        assert_eq!(
            ids(&compiled.report),
            ["eviction_interval", "max_idle_secs", "policy"]
        );
        assert!(compiled.report.consumed[1].unused.is_some());
        assert_eq!(compiled.check_against(&compiled.spec), Ok(()));
    }

    /// The prefix fields a side's policy flag implies stay on that side's
    /// row: a document that drops the decode side's is refused on the decode
    /// flag, and the prefill row carries none.
    #[test]
    fn the_implied_prefix_fields_stay_on_their_sides_row() {
        let compiled = compile_given(
            &pair(
                Some(cache_aware()),
                Some(PolicyConfig::PrefixHash {
                    prefix_token_count: 256,
                    load_factor: 1.25,
                    balance_abs_threshold: 10,
                    cache_boundaries: Vec::new(),
                }),
            ),
            &given(&["policy"]),
        );
        let row = |id: &str| {
            compiled
                .report
                .consumed
                .iter()
                .find(|row| row.id == id)
                .unwrap()
        };
        assert_eq!(
            paths(row("prefill_policy")),
            ["bindings[0].profiles.prefill", "profiles.prefill"]
        );
        assert_eq!(
            paths(row("decode_policy")),
            [
                "bindings[0].profiles.decode",
                "profiles.decode",
                "profiles.decode.score.consistent-hash.prefix_tokens",
                "profiles.decode.score.consistent-hash.bounded_load"
            ]
        );
        let mut document = serde_json::to_value(&compiled.spec).unwrap();
        document["profiles"]["decode"]["score"][0]["consistent-hash"]
            .as_object_mut()
            .unwrap()
            .remove("prefix_tokens");
        let plain: PolicySpec = serde_json::from_value(document).unwrap();
        let refused = compiled
            .check_against(&plain)
            .expect_err("the decode side's prefix form dropped");
        assert_eq!(refused.flag, "--decode-policy");
        assert_eq!(
            refused.field,
            "profiles.decode.score.consistent-hash.prefix_tokens"
        );
        assert_eq!(compiled.check_against(&compiled.spec), Ok(()));
    }

    /// A header list compares whole and in order: a document that widens or
    /// reorders the routing key headers is refused on the headers flag.
    #[test]
    fn a_header_list_widened_or_reordered_is_refused() {
        let config = RouterConfig::builder()
            .regular_mode(vec!["http://worker-a:8000".to_string()])
            .policy(PolicyConfig::RoundRobin)
            .routing_key_override(RoutingKeyOverrideConfig {
                enabled: true,
                eviction_interval_secs: 120,
                max_idle_secs: 14_400,
                assignment_mode: ManualAssignmentMode::Random,
                headers: vec!["x-session-id".to_string()],
            })
            .build_unchecked();
        let compiled = compile_given(&config, &given(&["routing_key_headers"]));
        for headers in [
            serde_json::json!(["x-session-id", "x-smg-routing-key"]),
            serde_json::json!(["x-smg-routing-key", "x-session-id"]),
        ] {
            let mut document = serde_json::to_value(&compiled.spec).unwrap();
            document["bindings"][0]["sessions"]["headers"] = headers;
            let changed: PolicySpec = serde_json::from_value(document).unwrap();
            let refused = compiled
                .check_against(&changed)
                .expect_err("the header list changed");
            assert_eq!(refused.flag, "--routing-key-headers");
            assert_eq!(refused.field, "bindings[0].sessions.headers");
        }
        assert_eq!(compiled.check_against(&compiled.spec), Ok(()));
    }

    /// A zero sweep keeps keys for ever, so the sweep flag owns both
    /// sessions fields and an idle time given beside it has its line with
    /// the reason.
    #[test]
    fn a_zero_sweep_owns_the_idle_field() {
        let manual = |max_idle_secs| PolicyConfig::Manual {
            eviction_interval_secs: 0,
            max_idle_secs,
            assignment_mode: ManualAssignmentMode::Random,
        };
        let compiled = compile(&regular(manual(14_400)));
        assert_eq!(ids(&compiled.report), ["eviction_interval", "policy"]);
        let sweep = &compiled.report.consumed[0];
        assert_eq!(
            paths(sweep),
            ["bindings[0].sessions.sweep_s", "bindings[0].sessions.ttl_s"]
        );
        assert_eq!(compiled.check_against(&compiled.spec), Ok(()));

        let carried = compile_given(&regular(manual(600)), &given(&["max_idle_secs"]));
        assert_eq!(
            ids(&carried.report),
            ["eviction_interval", "max_idle_secs", "policy"]
        );
        assert_eq!(
            carried.report.consumed[1].deprecation_line("1.17", false),
            "smg: --max-idle-secs is deprecated and is removed in 1.17; it compiled to \
             nothing (a zero sweep keeps keys for ever, so the idle time is for ever too); \
             write it in --policy-spec"
        );
        assert_eq!(carried.check_against(&carried.spec), Ok(()));
    }

    /// A policy flag is held by a document whose profile has the policy's
    /// stages; a profile of another shape under the same name is refused
    /// naming the field and the shape held.
    #[test]
    fn a_profile_of_another_shape_refuses_the_policy_flag() {
        let compiled = compile_given(&regular(cache_aware()), &given(&["policy"]));
        let other = doc("\
apiVersion: smg.policy/v1
generation: 1
profiles:
  cache_aware:
    filter: [healthy, not-overloaded, not-tried]
    score: [least-load]
    pick: best
bindings:
  - match: { model: \"*\" }
    profile: cache_aware
");
        let refused = compiled
            .check_against(&other)
            .expect_err("another shape under the name");
        assert_eq!(refused.field, "profiles.cache_aware");
        assert_eq!(
            refused.held,
            "filter [healthy, not-overloaded, not-tried], score [least-load], pick best"
        );
    }

    /// Every document the converter writes validates under the engine, or
    /// is refused only for a stage, a plan or a binding rule this build
    /// lacks: never for its shape.
    #[test]
    fn every_compiled_document_validates_or_names_a_stage_this_build_lacks() {
        let configs = [
            regular(cache_aware()),
            regular(least_load()),
            regular(PolicyConfig::RoundRobin),
            regular(PolicyConfig::ConsistentHashing),
            regular(PolicyConfig::Manual {
                eviction_interval_secs: 30,
                max_idle_secs: 3600,
                assignment_mode: ManualAssignmentMode::MinGroup,
            }),
            regular(PolicyConfig::cache_aware(0.5, 32, 1.1, 60, 16_777_216)),
            pair(
                Some(PolicyConfig::RoundRobin),
                Some(PolicyConfig::RoundRobin),
            ),
            pair(None, None),
            pair(
                Some(PolicyConfig::Bucket {
                    balance_abs_threshold: 64,
                    balance_rel_threshold: 1.5,
                    bucket_adjust_interval_secs: 5,
                }),
                Some(PolicyConfig::PowerOfTwo {
                    load_metric: PowerOfTwoLoadMetric::ExpectedWait,
                }),
            ),
            regular(PolicyConfig::PrefixHash {
                prefix_token_count: 256,
                load_factor: 1.25,
                balance_abs_threshold: 10,
                cache_boundaries: Vec::new(),
            }),
            regular(PolicyConfig::PowerOfTwo {
                load_metric: PowerOfTwoLoadMetric::ResidentTokens {
                    request_tokens: 4096,
                },
            }),
            regular(PolicyConfig::Manual {
                eviction_interval_secs: 60,
                max_idle_secs: 0,
                assignment_mode: ManualAssignmentMode::Random,
            }),
            RouterConfig::builder()
                .regular_mode(vec!["http://worker-a:8000".to_string()])
                .policy(cache_aware())
                .routing_key_override(RoutingKeyOverrideConfig {
                    enabled: true,
                    eviction_interval_secs: 60,
                    max_idle_secs: 0,
                    assignment_mode: ManualAssignmentMode::Delegate,
                    headers: vec!["x-session-id".to_string()],
                })
                .build_unchecked(),
        ];
        for config in &configs {
            let compiled = compile(config);
            match compiled.spec.validate() {
                Ok(_) | Err(SpecError::NotInBuild { .. }) => {}
                Err(other) => panic!("{other}: {:?}", compiled.spec),
            }
            // The round trip: every flag the converter consumed is held by
            // the document it wrote, whatever the flag set.
            assert_eq!(
                compiled.check_against(&compiled.spec),
                Ok(()),
                "{:?}",
                compiled.report
            );
            let every: GivenFlags = FLAGS.iter().map(|flag| flag.id.to_string()).collect();
            let given = compile_given(config, &every);
            assert_eq!(
                given.check_against(&given.spec),
                Ok(()),
                "{:?}",
                given.report
            );
        }
    }

    /// A front end that marks its given set by keyword gives both flags of
    /// the overload switch; the one the configuration consumes rows, and
    /// the other is not an unused flag beside it.
    #[test]
    fn a_keyword_two_flags_share_rows_once_when_given() {
        let mut config = regular(cache_aware());
        config.worker_overload_protection = false;
        let compiled = compile_given(
            &config,
            &given(&[
                "worker_overload_protection",
                "disable_worker_overload_protection",
            ]),
        );
        let rows: Vec<&str> = compiled
            .report
            .consumed
            .iter()
            .filter(|row| row.keyword == "worker_overload_protection")
            .map(|row| row.id)
            .collect();
        assert_eq!(rows, ["disable_worker_overload_protection"]);
        assert_eq!(compiled.check_against(&compiled.spec), Ok(()));
    }

    fn prefix_hash_given(ids: &[&str]) -> Compiled {
        compile_given(
            &regular(PolicyConfig::PrefixHash {
                prefix_token_count: 256,
                load_factor: 1.25,
                balance_abs_threshold: 10,
                cache_boundaries: Vec::new(),
            }),
            &given(ids),
        )
    }

    /// The bound is read from its block: an empty block is the bound with
    /// the engine's defaults and passes under the bare policy flag; the
    /// block left out is plain hashing and is refused on it.
    #[test]
    fn an_empty_bound_block_is_the_bound_under_the_bare_prefix_flag() {
        let compiled = prefix_hash_given(&["policy"]);
        let mut document = serde_json::to_value(&compiled.spec).unwrap();
        let hash = &mut document["profiles"]["prefix_hash"]["score"][0]["consistent-hash"];
        hash["bounded_load"] = serde_json::json!({});
        let bare: PolicySpec = serde_json::from_value(document.clone()).unwrap();
        assert_eq!(compiled.check_against(&bare), Ok(()));
        document["profiles"]["prefix_hash"]["score"][0]["consistent-hash"]
            .as_object_mut()
            .unwrap()
            .remove("bounded_load");
        let unbounded: PolicySpec = serde_json::from_value(document).unwrap();
        let refused = compiled
            .check_against(&unbounded)
            .expect_err("the bound dropped");
        assert_eq!(refused.flag, "--policy");
        assert_eq!(
            refused.field,
            "profiles.prefix_hash.score.consistent-hash.bounded_load"
        );
        assert_eq!(refused.held, "nothing");
    }

    /// A knob given at its default keeps the policy's mark: the field left
    /// out is refused on the policy flag, another value on the knob's.
    #[test]
    fn a_knob_given_at_its_default_keeps_the_prefix_mark() {
        let compiled = prefix_hash_given(&["policy", "prefix_token_count"]);
        let path = "profiles.prefix_hash.score.consistent-hash.prefix_tokens";
        let mut document = serde_json::to_value(&compiled.spec).unwrap();
        document["profiles"]["prefix_hash"]["score"][0]["consistent-hash"]["prefix_tokens"] =
            serde_json::json!(512);
        let longer: PolicySpec = serde_json::from_value(document.clone()).unwrap();
        let refused = compiled
            .check_against(&longer)
            .expect_err("the knob's value moved");
        assert_eq!(
            (refused.flag, refused.field.as_str()),
            ("--prefix-token-count", path)
        );
        document["profiles"]["prefix_hash"]["score"][0]["consistent-hash"]
            .as_object_mut()
            .unwrap()
            .remove("prefix_tokens");
        let plain: PolicySpec = serde_json::from_value(document).unwrap();
        let refused = compiled
            .check_against(&plain)
            .expect_err("the prefix form dropped");
        assert_eq!((refused.flag, refused.field.as_str()), ("--policy", path));
    }

    /// The session ring takes neither a prefix key nor a bound: a document
    /// writing one under the name routes another way and is refused.
    #[test]
    fn a_prefix_key_under_the_session_ring_is_refused() {
        let compiled = compile_given(
            &regular(PolicyConfig::ConsistentHashing),
            &given(&["policy"]),
        );
        assert_eq!(compiled.check_against(&compiled.spec), Ok(()));
        let mut document = serde_json::to_value(&compiled.spec).unwrap();
        let profiles = document["profiles"].as_object_mut().unwrap();
        let name = profiles.keys().next().unwrap().clone();
        profiles[&name]["score"] =
            serde_json::json!([{ "consistent-hash": { "prefix_tokens": 512 } }]);
        let keyed: PolicySpec = serde_json::from_value(document).unwrap();
        let refused = compiled
            .check_against(&keyed)
            .expect_err("a prefix key under the ring");
        assert_eq!(refused.flag, "--policy");
        assert_eq!(
            refused.field,
            format!("profiles.{name}.score.consistent-hash.prefix_tokens")
        );
        assert_eq!(refused.held, "512");
    }

    /// A refusal names the flag's value as typed: the metric's spelling on
    /// its multi-field row, the idle seconds the schema writes as for ever.
    #[test]
    fn a_refusal_shows_the_flag_as_typed() {
        let compiled = compile_given(
            &regular(PolicyConfig::PowerOfTwo {
                load_metric: PowerOfTwoLoadMetric::ResidentTokens {
                    request_tokens: 4096,
                },
            }),
            &given(&["policy", "power_of_two_load_metric"]),
        );
        let mut document = serde_json::to_value(&compiled.spec).unwrap();
        document["profiles"]["power_of_two"]["score"][0]["resident-work"]["request_tokens"] =
            serde_json::json!(2048);
        let shorter: PolicySpec = serde_json::from_value(document).unwrap();
        let refused = compiled
            .check_against(&shorter)
            .expect_err("the request tokens moved");
        assert_eq!(refused.flag, "--power-of-two-load-metric");
        assert_eq!(refused.shown, "resident_tokens:4096");
        assert_eq!(
            refused.field,
            "profiles.power_of_two.score.resident-work.request_tokens"
        );
        assert_eq!(refused.held, "2048");

        let compiled = compile_given(
            &regular(PolicyConfig::Manual {
                eviction_interval_secs: 120,
                max_idle_secs: 0,
                assignment_mode: ManualAssignmentMode::Random,
            }),
            &given(&["policy", "max_idle_secs"]),
        );
        let mut document = serde_json::to_value(&compiled.spec).unwrap();
        document["bindings"][0]["sessions"]["ttl_s"] = serde_json::json!(600);
        let bounded: PolicySpec = serde_json::from_value(document).unwrap();
        let refused = compiled
            .check_against(&bounded)
            .expect_err("the idle time moved");
        assert_eq!(
            (refused.flag, refused.shown.as_str()),
            ("--max-idle-secs", "0")
        );
        assert_eq!(refused.field, "bindings[0].sessions.ttl_s");
    }

    /// A zero idle time keeps keys for ever: a sweep off its default beside
    /// it is unused and has its row with the reason, a given set or not.
    #[test]
    fn a_sweep_off_default_beside_a_zero_idle_time_has_its_row_without_a_given_set() {
        let compiled = compile(&regular(PolicyConfig::Manual {
            eviction_interval_secs: 7,
            max_idle_secs: 0,
            assignment_mode: ManualAssignmentMode::Random,
        }));
        assert_eq!(
            ids(&compiled.report),
            ["max_idle_secs", "eviction_interval", "policy"]
        );
        let sweep = &compiled.report.consumed[1];
        assert!(sweep.fields.is_empty());
        assert_eq!(sweep.shown, "7");
        assert_eq!(
            sweep.unused.as_deref(),
            Some("a zero idle time keeps keys for ever, so the sweep is for ever too")
        );
        assert_eq!(compiled.check_against(&compiled.spec), Ok(()));
    }
}
