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
    /// The document fields, by path
    /// (`profiles.cache_aware.score.prefix-affinity.threshold`); empty for
    /// a flag the routing mode leaves unused.
    pub fields: Vec<String>,
    /// The value as the deprecation line shows it.
    pub shown: String,
    /// The value as the document renders it.
    pub value: Value,
    /// The value the flag has when it is left out, rendered the same way;
    /// `null` for a flag with no default.
    pub default: Value,
    /// Whether the value is the default: the row exists because the flag
    /// was given.
    pub at_default: bool,
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

    /// "`<fields> = <value>`", or what became of a flag with no field.
    fn landing(&self, spec_holds: bool) -> String {
        if self.fields.is_empty() {
            return format!("nothing ({} routes no request in this mode)", self.shown);
        }
        let holds = if spec_holds {
            ", which the spec already holds"
        } else {
            ""
        };
        format!("{} = {}{holds}", self.fields.join(" and "), self.shown)
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
    }
    .run()
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
            ..
        } = self;
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

    /// The regular modes: one profile named after the policy, bound to
    /// every model.
    fn single(&mut self) {
        let config = self.config;
        let policy = &config.policy;
        let name = policy.name();
        let profile = self.profile(policy, Side::Single, name);
        self.profiles.insert(name.to_string(), profile);
        self.binding.profile = Some(name.to_string());
        self.row(
            POLICY,
            vec![
                "bindings[0].profile".to_string(),
                format!("profiles.{name}"),
            ],
            &name,
            Some(&DEFAULT_POLICY),
        );
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
        self.profiles.insert("prefill".to_string(), prefill_profile);
        self.profiles.insert("decode".to_string(), decode_profile);
        let mut sides = BTreeMap::new();
        sides.insert("prefill".to_string(), "prefill".to_string());
        sides.insert("decode".to_string(), "decode".to_string());
        let mut phases = vec![Phase::Prefill, Phase::Decode];
        let mut plan = PlanName::PdPair;
        if let Some(encode) = epd {
            let policy = encode.unwrap_or(&DEFAULT_ENCODE_POLICY);
            let encode_profile = self.profile(policy, Side::Encode, "encode");
            self.profiles.insert("encode".to_string(), encode_profile);
            sides.insert("encode".to_string(), "encode".to_string());
            phases.insert(0, Phase::Encode);
            plan = PlanName::Epd;
            match encode {
                Some(policy) => self.row(
                    ENCODE_POLICY,
                    vec![
                        "bindings[0].profiles.encode".to_string(),
                        "profiles.encode".to_string(),
                    ],
                    &policy.name(),
                    None,
                ),
                None => self.notes.push(format!(
                    "no --encode-policy given: the default encode profile {} was written",
                    DEFAULT_ENCODE_POLICY.name()
                )),
            }
        }
        let mut main_fields = Vec::new();
        match prefill {
            Some(policy) => self.row(
                PREFILL_POLICY,
                vec![
                    "bindings[0].profiles.prefill".to_string(),
                    "profiles.prefill".to_string(),
                ],
                &policy.name(),
                None,
            ),
            None => main_fields.push("profiles.prefill".to_string()),
        }
        match decode {
            Some(policy) => self.row(
                DECODE_POLICY,
                vec![
                    "bindings[0].profiles.decode".to_string(),
                    "profiles.decode".to_string(),
                ],
                &policy.name(),
                None,
            ),
            None => main_fields.push("profiles.decode".to_string()),
        }
        if main_fields.is_empty() {
            self.notes.push(format!(
                "the main policy {} routes no request under {mode_flag}: it compiled to nothing",
                main.name()
            ));
        }
        self.row(POLICY, main_fields, &main.name(), Some(&DEFAULT_POLICY));
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
        let (score, pick): (ScoreSpec, &str) = match metric {
            PowerOfTwoLoadMetric::ExpectedWait => ("least-load".into(), "power-of-two"),
            PowerOfTwoLoadMetric::Requests => ("least-load".into(), "power-of-two-count-first"),
            PowerOfTwoLoadMetric::LeastRequests => ("least-load".into(), "count-first"),
            PowerOfTwoLoadMetric::ResidentTokens { request_tokens } => (
                ScoreSpec::Params(ScoreStage::ResidentWork(ResidentWorkParams {
                    weight: 1.0,
                    request_tokens: saturating_u32(request_tokens),
                })),
                "best",
            ),
        };
        self.row(
            POWER_OF_TWO_LOAD_METRIC,
            vec![format!("{p}.pick")],
            &pick,
            Some(&"power-of-two"),
        );
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
        let bounded = BoundedLoadParams {
            factor: off(factor, 1.25),
            abs_margin: off(margin, 10),
        };
        let params = ConsistentHashParams {
            weight: 1.0,
            prefix_tokens: off(prefix_tokens, 256),
            bounded_load: (bounded != BoundedLoadParams::default()).then_some(bounded),
        };
        ProfileSpec {
            filter: side.filters(&STANDARD),
            score: vec![stage(
                params != ConsistentHashParams::default(),
                ScoreStage::ConsistentHash(params),
                "consistent-hash",
            )],
            pick: "best".into(),
        }
    }

    /// The manual policy's sessions: its key header, its idle time and
    /// sweep, how a first-seen key is placed, the cap of two in flight.
    fn manual_sessions(&mut self, sweep: u64, idle: u64, mode: ManualAssignmentMode) {
        let (ttl_s, sweep_s) = lifetime(idle, sweep);
        self.row(
            MAX_IDLE_SECS,
            vec!["bindings[0].sessions.ttl_s".to_string()],
            &ttl_s,
            Some(&14_400_u32),
        );
        self.row(
            EVICTION_INTERVAL,
            vec!["bindings[0].sessions.sweep_s".to_string()],
            &sweep_s,
            Some(&120_u32),
        );
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
        self.row(
            MAX_IDLE_SECS,
            vec!["bindings[0].sessions.ttl_s".to_string()],
            &ttl_s,
            Some(&14_400_u32),
        );
        self.row(
            EVICTION_INTERVAL,
            vec!["bindings[0].sessions.sweep_s".to_string()],
            &sweep_s,
            Some(&120_u32),
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
        let rendered = value.value();
        let (default, at_default) = match default {
            Some(default) => {
                let default = default.value();
                let at_default = default == rendered;
                (default, at_default)
            }
            None => (Value::Null, false),
        };
        if at_default && !self.given.contains(flag.id) {
            return;
        }
        if let Some(row) = self.consumed.iter_mut().find(|row| row.id == flag.id) {
            for field in fields {
                if !row.fields.contains(&field) {
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
            shown: value.shown(),
            value: rendered,
            default,
            at_default,
        });
    }
}

impl Compiled {
    /// The conflict rule: every consumed flag against the
    /// `--policy-spec` document. `Ok` when the document holds each flag's
    /// value at its field (a field the document leaves out holds the
    /// default), the first flag whose value it does not hold otherwise.
    /// The policy names are checked first, so a document that names
    /// another profile is refused on `--policy` and not on a knob under the
    /// profile it does not have; a flag the mode leaves unused is not
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
                let found = lookup(&document, field);
                let agrees = match found {
                    Some(found) => holds(found, &row.value),
                    None => same(&row.default, &row.value),
                };
                if !agrees {
                    return Err(Disagreement {
                        flag: row.flag,
                        shown: row.shown.clone(),
                        field: field.clone(),
                        held: found.map_or_else(|| "nothing".to_string(), render),
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
            sweep.fields,
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
        let unused = both.report.deprecation_lines("1.17", false)[2].clone();
        assert!(
            unused.contains("it compiled to nothing (cache_aware routes no request in this mode)"),
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
        assert_eq!(main.fields, ["profiles.prefill", "profiles.decode"]);
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
        ];
        for config in &configs {
            let compiled = compile(config);
            match compiled.spec.validate() {
                Ok(_) | Err(SpecError::NotInBuild { .. }) => {}
                Err(other) => panic!("{other}: {:?}", compiled.spec),
            }
        }
    }
}
