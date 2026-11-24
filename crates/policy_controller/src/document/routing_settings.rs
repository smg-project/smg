//! The policy document a gateway's routing settings mean: [`RoutingSettings`],
//! the types it is made of, and why some settings have no document
//! ([`RoutingSettingsError`]).

use serde::Serialize;
use serde_json::Value;

use crate::{
    document::spec::{
        Assignment, BestWithTemperatureParams, BindingSpec, BoundedLoadParams, BucketParams,
        ConsistentHashParams, IndexKind, IndexSpec, KeySourceSpec, KvPressureParams,
        LeastLoadParams, OverloadSpec, Pairing, PickStage, PolicySpec, PrefixAffinityParams,
        ProfileSpec, Recovery, RoutingKeySpec, RulePlaced, ScoreSpec, ScoreStage, SessionsSpec,
        SpecError, SpillParams, WarmupSliceParams, MANUAL_SESSION_HEADER, MAX_KEY_SOURCES,
        OVERLOAD_WAITING_REQUESTS,
    },
    engine::view::Phase,
    policies::{
        bucket,
        cache_aware::{
            Gates, DEFAULT_THRESHOLD, DEFAULT_WARMUP_BLOCKS, DEFAULT_WARMUP_DIVERT_EVERY,
            DEFAULT_WARMUP_SECS, DEFAULT_WARMUP_SHARE, DEFAULT_WARMUP_THIN_RATIO,
        },
        consistent_hashing::target_source,
        power_of_two::PowerOfTwoMetric,
    },
    stages::scorers::{
        DEFAULT_KV_PRESSURE_WEIGHT, DEFAULT_MEAN_PREFILL_TOKENS, DEFAULT_THROUGHPUT,
    },
};

/// A gateway's routing settings, and the policy document they mean.
///
/// # Why this exists
///
/// A gateway that ran routing policies of its own holds routing settings: a
/// policy by its name with its numbers, a mode, and a few settings every
/// policy shares. This type is those settings in the one form this crate
/// reads: what the gateway's flags, its configuration file or its bindings say
/// about where a request goes. The gateway maps its own configuration onto it.
/// A field carries the name the configuration gives its setting, and a
/// refusal names a setting by that name.
///
/// [`RoutingSettings::policy_document`] converts the settings into the
/// controller's [`PolicySpec`]. [`RoutingSettings::policy_document_in_force`]
/// takes a supplied document when the settings are at the gateway's defaults
/// or describe that same document. A gateway calls them at its start, and
/// applies the document in force to its store. No request path calls them.
///
/// # Where a setting lands
///
/// - A policy name lands on the controller's preset of that name. A
///   prefill-decode mode lands on the pair plan, with one profile a side.
/// - A gateway in the prefill-decode mode places one single request beside
///   its pairs over HTTP, a token count that takes one prefill worker, and
///   a gateway of many routers serves pairs and single workers over its one
///   document. Either document binds both: pairs first, over the prefill
///   and decode phases, and single requests behind. The sides of a pair run
///   under the mode's side policies and single requests under the main
///   policy, as under the gateway's own policies. Over gRPC or ZMQ the mode
///   counts tokens with no worker, so a gateway of one router plans the
///   pair alone there ([`ConnectionModeSetting`]).
/// - A policy's own setting lands in the stage of the preset that reads it,
///   and only where it differs from the number that stage runs at when its
///   document names none. The controller names those numbers and this
///   module reads them, so a configuration at them is the preset itself.
/// - The overload veto's switch and its shed word land on every binding,
///   the pairing mode on the pair plan, the index settings under every
///   model, and the block size and the cache boundaries in the view.
/// - The manual policy's settings land in the binding's session rule.
/// - A keyed policy's key sources land on the binding. The two rings name
///   them under `routing_key`. The manual policy names them in its session
///   rule while they are headers alone, and under `routing_key` once the
///   override adds the request's own id.
/// - The sticky override lands as a session rule on the binding of
///   `random`, `power_of_two`, `least_load`, `passthrough`, `bucket` and
///   `cache_aware` (`override_places` says why these six):
///   the override's own assignment and times, its two requests of a key a
///   worker, and the configured headers then the request's own id under
///   `routing_key`. A rule with a cap runs under the single plan alone,
///   so it has a document in a gateway of one router that serves single
///   requests.
/// - A number that means never lands as the largest its place holds: a
///   balance threshold past the document's width, and the sweep interval of
///   an index the configuration never sweeps. The gateway's own policy
///   trims an index at its sweep alone, so that index's size bound lands as
///   the largest too.
///
/// # What is refused
///
/// A setting that decides a placement and has no place in the document is
/// refused by its name ([`RoutingSettingsError::UnsupportedSetting`]). It is never left
/// out in silence, but for what the gateway keeps for itself (below).
///
/// # What is left out
///
/// A document of two families plans one where the other has no document.
/// Pairs are left out of a regular configuration of many routers under the
/// manual policy, whose pairs this module does not plan beside its single
/// requests yet, and beside the encode mode. Single requests are left out
/// of a pair gateway of one router that reaches its workers over gRPC or
/// ZMQ: it places none.
/// They are left out of a pair configuration whose main policy reads a
/// setting no document says: a gateway of one router then refuses its token
/// count, which its own policies place. That gateway leaves them out as well
/// where one family's policy names a block size and the other places by a
/// prefix ring, which counts its key in another size. A request of the
/// family left out is refused.
///
/// # What is not read
///
/// What the gateway keeps for itself under the controller is no part of the
/// document: the pools and their addresses, discovery, the probes and the
/// load poll, retries and the breaker, the prefill queue and the decode
/// rooms. The two overload thresholds are the gateway's too: the default it
/// builds its worker objects with, under each worker's own registration. No
/// document carries a threshold, so one off its default means the same
/// document. Both at none leave nothing to veto by, and the settings' own
/// document switches the block off. Beside a supplied document the two are
/// no difference, whatever they are: its host says what a block that is on
/// means over no threshold. The pin of a sticky key to one engine of a
/// grouped worker is the dispatch's as well: it runs behind the decision, on
/// the worker the decision names. Its key is the gateway's own, read from
/// the routing-key headers. A supplied document names its own key sources,
/// which need not be that key, so the pin is refused beside one. The copy
/// of the cache boundaries at the top of the configuration decides nothing
/// either: each policy reads its own.
#[derive(Clone, Debug, PartialEq)]
pub struct RoutingSettings {
    /// The kinds of worker the gateway routes to, with each side's own policy.
    pub mode: ModeSettings,
    /// The main policy. Single workers run under it, and so does a side of
    /// the mode that names no policy of its own.
    pub policy: PolicySettings,
    /// The gateway runs many routers side by side, so one document plans pairs
    /// and single workers.
    pub enable_igw: bool,
    /// How the gateway reaches its workers. A pair mode of one router places
    /// a single request, its token count, over HTTP alone.
    pub connection_mode: ConnectionModeSetting,
    /// A document supplied beside the settings, to run in place of the one
    /// they mean.
    pub policy_spec: Option<PolicySpec>,
    /// The pin of a sticky key to one engine of a grouped worker: its
    /// spelling in the configuration where the pin is on, and none where it
    /// is off.
    pub grouped_dp_sticky: Option<&'static str>,
    /// The gateway chooses a data-parallel rank itself, from the ranks'
    /// reported token loads.
    pub dp_minimum_tokens_scheduler: bool,
    /// How strictly a prefill pairs with a decode.
    pub pd_pairing_mode: Pairing,
    /// The warm-up slice of the cache-aware policy: the seconds a worker
    /// stays in it after it becomes routable.
    pub worker_warmup_secs: u64,
    /// The share of cache misses offered to the workers of the slice.
    pub worker_warmup_share: f32,
    /// The blocks a worker's index grows by before it leaves the slice.
    pub worker_warmup_blocks: usize,
    /// The share of the fleet's level under which a worker's index is thin.
    pub worker_warmup_thin_ratio: f32,
    /// One cache hit in this many is diverted to a thin worker.
    pub worker_warmup_divert_every: u64,
    /// The sticky override, and the headers a request's routing key is read
    /// from.
    pub routing_key_override: RoutingKeyOverrideSettings,
    /// The overload veto is on.
    pub worker_overload_protection: bool,
    /// The waiting requests at which a worker counts as overloaded. None
    /// switches the signal off. The gateway's own: no document carries it.
    pub worker_overload_waiting_requests: Option<usize>,
    /// The KV-cache token usage at which a worker counts as overloaded. None
    /// switches the signal off. The gateway's own: no document carries it.
    pub worker_overload_token_usage: Option<f64>,
    /// A request is refused where every worker it could use is overloaded.
    pub worker_overload_shed: bool,
}

impl Default for RoutingSettings {
    /// The settings of a gateway that sets none: each setting at the value it
    /// has where a configuration leaves it out. The document they mean is the
    /// `random` preset.
    ///
    /// The warm-up numbers are the ones the controller's stages run at when
    /// a document names none, so settings at them write none into a
    /// document. The overload thresholds are the crate's two defaults, and
    /// no document carries one at any value.
    fn default() -> Self {
        Self {
            mode: ModeSettings::Regular,
            policy: PolicySettings::Random,
            enable_igw: false,
            connection_mode: ConnectionModeSetting::Http,
            policy_spec: None,
            grouped_dp_sticky: None,
            dp_minimum_tokens_scheduler: false,
            pd_pairing_mode: Pairing::Lenient,
            worker_warmup_secs: DEFAULT_WARMUP_SECS,
            worker_warmup_share: DEFAULT_WARMUP_SHARE,
            worker_warmup_blocks: DEFAULT_WARMUP_BLOCKS as usize,
            worker_warmup_thin_ratio: DEFAULT_WARMUP_THIN_RATIO,
            worker_warmup_divert_every: DEFAULT_WARMUP_DIVERT_EVERY,
            routing_key_override: RoutingKeyOverrideSettings::default(),
            worker_overload_protection: true,
            worker_overload_waiting_requests: Some(OVERLOAD_WAITING_REQUESTS as usize),
            // A configuration writes this usage as 0.8 in 64 bits. The
            // controller's number is 32 bits wide: widened, it is another
            // value.
            worker_overload_token_usage: Some(0.8),
            worker_overload_shed: false,
        }
    }
}

impl RoutingSettings {
    /// The provider's name, where every router of the gateway is that
    /// provider's: a provider mode of a gateway that does not run many routers.
    /// It places by no routing policy: a host asks here ahead of the document.
    pub fn provider_alone(&self) -> Option<&'static str> {
        match self.mode {
            ModeSettings::Provider(provider) => (!self.enable_igw).then_some(provider),
            ModeSettings::Regular
            | ModeSettings::PrefillDecode { .. }
            | ModeSettings::EncodePrefillDecode { .. } => None,
        }
    }
}

/// How a gateway reaches its workers, as the configuration names it. One
/// thing follows from it here: whether a gateway of one router in the pair
/// mode places a single request, which its document then binds
/// ([`RoutingSettings`], where a setting lands).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConnectionModeSetting {
    /// Over HTTP. A pair mode counts tokens on one prefill worker, a single
    /// request its own routing places by the main policy.
    Http,
    /// Over gRPC. A pair mode counts tokens with no worker.
    Grpc,
    /// Over ZMQ, by the router that serves gRPC: no worker counts tokens.
    Zmq,
}

/// The settings of the sticky override this module reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutingKeyOverrideSettings {
    /// A request that carries a sticky key is placed by the sticky sessions'
    /// rule, whatever the policy.
    pub enabled: bool,
    /// The header names a request's routing key is read from, in order.
    pub headers: Vec<String>,
    /// The seconds between two sweeps of the override's sessions. Zero here
    /// or in `max_idle_secs` keeps every key the session table holds pinned
    /// for ever: a table full of such keys records no new key.
    pub eviction_interval_secs: u64,
    /// The seconds with no request after which a key of the override leaves
    /// at the next sweep. Zero here or in `eviction_interval_secs` keeps
    /// every key the session table holds pinned for ever: a table full of
    /// such keys records no new key.
    pub max_idle_secs: u64,
    /// How a key the override sees for the first time picks its worker.
    pub assignment_mode: Assignment,
}

impl Default for RoutingKeyOverrideSettings {
    /// The override of a gateway that sets none of it: off, with the header,
    /// the two times and the assignment a configuration that leaves them out
    /// gets.
    fn default() -> Self {
        Self {
            enabled: false,
            headers: vec![MANUAL_SESSION_HEADER.to_owned()],
            eviction_interval_secs: 60,
            max_idle_secs: 4 * 3600,
            assignment_mode: Assignment::Delegate,
        }
    }
}

/// The kinds of worker a gateway routes to. A side names its own policy or
/// runs under the main one. The pools of a mode and their addresses are the
/// gateway's own, and no part of this type.
#[derive(Clone, Debug, PartialEq)]
pub enum ModeSettings {
    /// Single workers.
    Regular,
    /// Pairs of a prefill worker and a decode worker.
    PrefillDecode {
        /// The prefill side's own policy.
        prefill_policy: Option<PolicySettings>,
        /// The decode side's own policy.
        decode_policy: Option<PolicySettings>,
    },
    /// An encode worker ahead of a pair.
    EncodePrefillDecode {
        /// The encode side's own policy.
        encode_policy: Option<PolicySettings>,
        /// The prefill side's own policy.
        prefill_policy: Option<PolicySettings>,
        /// The decode side's own policy.
        decode_policy: Option<PolicySettings>,
    },
    /// The workers of an outside provider, under the mode's name in the
    /// configuration.
    Provider(&'static str),
}

impl ModeSettings {
    /// The prefill side's policy: its own, or the main one.
    fn prefill_policy<'a>(&'a self, main: &'a PolicySettings) -> &'a PolicySettings {
        match self {
            Self::PrefillDecode { prefill_policy, .. }
            | Self::EncodePrefillDecode { prefill_policy, .. } => {
                prefill_policy.as_ref().unwrap_or(main)
            }
            Self::Regular | Self::Provider(_) => main,
        }
    }

    /// The decode side's policy: its own, or the main one.
    fn decode_policy<'a>(&'a self, main: &'a PolicySettings) -> &'a PolicySettings {
        match self {
            Self::PrefillDecode { decode_policy, .. }
            | Self::EncodePrefillDecode { decode_policy, .. } => {
                decode_policy.as_ref().unwrap_or(main)
            }
            Self::Regular | Self::Provider(_) => main,
        }
    }

    /// The encode side's policy: its own, or `unnamed`. The main policy is
    /// never the encode side's.
    fn encode_policy<'a>(&'a self, unnamed: &'a PolicySettings) -> &'a PolicySettings {
        match self {
            Self::EncodePrefillDecode { encode_policy, .. } => {
                encode_policy.as_ref().unwrap_or(unnamed)
            }
            Self::Regular | Self::PrefillDecode { .. } | Self::Provider(_) => unnamed,
        }
    }
}

/// What the `power_of_two` policy compares its candidates by, as the
/// configuration says it: [`PowerOfTwoMetric`], with the price of a request
/// in the configuration's width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadMetricSetting {
    /// The expected wait of two random workers.
    ExpectedWait,
    /// The requests in flight of two random workers.
    Requests,
    /// The requests in flight of every worker.
    LeastRequests,
    /// The resident context tokens of every worker, each request also priced
    /// at `request_tokens`.
    ResidentTokens {
        /// The price of one request, in context tokens.
        request_tokens: u64,
    },
}

/// A routing policy under its name in the configuration, with its settings.
/// A policy lands on the preset of its name ([`PolicySettings::name`]), and
/// each setting in the stage of that preset that reads it.
#[derive(Clone, Debug, PartialEq)]
pub enum PolicySettings {
    /// `random`.
    Random,
    /// `round_robin`.
    RoundRobin,
    /// `passthrough`: one worker, and no balancing.
    Passthrough,
    /// `cache_aware`: a request goes to the worker that holds its prefix,
    /// unless the load says otherwise.
    CacheAware {
        /// The matched share of a prompt at which a request pins to the
        /// worker that holds it.
        cache_threshold: f32,
        /// The spill gate's absolute part: requests over the fleet's mean.
        balance_abs_threshold: usize,
        /// The spill gate's relative part: a multiple of the fleet's mean.
        balance_rel_threshold: f32,
        /// The price of a held prefix at the spill gate, in tokens a request.
        balance_prefix_margin_tokens: u32,
        /// A pick on a holder is credited with the prompt it still has to
        /// prefill, and not with the whole prompt.
        balance_uncached_credit: bool,
        /// The seconds between two sweeps of the index. Zero sweeps nothing.
        eviction_interval_secs: u64,
        /// The size the index is trimmed to at a sweep.
        max_tree_size: usize,
        /// The tokens of one block.
        block_size: usize,
        /// The spread of KV usage over the fleet past which affinity is
        /// given up.
        balance_token_usage_threshold: f32,
        /// The KV usage of one worker past which affinity is given up.
        overload_token_usage_threshold: f32,
        /// The decay of an overlap score by the worker's waiting prefill.
        overlap_decay: f32,
        /// The temperature of the pick. Zero takes the best candidate.
        selection_temperature: f32,
        /// The under-layer of the index.
        cache_index: IndexKind,
        /// The seconds a placement stays in the index.
        cache_ttl_secs: u64,
        /// The token positions a request's head is keyed at.
        cache_boundaries: Vec<usize>,
        /// A selection policy other than the gateway's default one. The gateway
        /// leaves its default's name out: no document says another.
        selection_policy: Option<String>,
        /// The milliseconds an optimistic booking lives. Zero books none.
        selection_accounting_ttl_ms: u64,
    },
    /// `power_of_two`.
    PowerOfTwo {
        /// What the policy compares its candidates by.
        load_metric: LoadMetricSetting,
    },
    /// `least_load`: the lowest expected wait.
    LeastLoad {
        /// The weight of KV pressure, in seconds.
        kv_pressure_weight: f64,
        /// The prefill length assumed for a request of unknown size, in
        /// tokens.
        mean_prefill_tokens: u32,
        /// The throughput of a prefill worker, and of any worker where none
        /// reports its own, in tokens a second.
        default_throughput: f64,
    },
    /// `bucket`.
    Bucket {
        /// The absolute load difference that starts balancing.
        balance_abs_threshold: usize,
        /// The relative load ratio that starts balancing.
        balance_rel_threshold: f32,
        /// The seconds between two adjustments of the bucket boundaries.
        bucket_adjust_interval_secs: usize,
    },
    /// `manual`: sticky sessions by a routing key.
    Manual {
        /// The seconds between two sweeps of the sessions.
        eviction_interval_secs: u64,
        /// The seconds a key stays pinned with no request.
        max_idle_secs: u64,
        /// How a key seen for the first time picks its worker.
        assignment_mode: Assignment,
    },
    /// `consistent_hashing`.
    ConsistentHashing,
    /// `prefix_hash`: a hash ring over a prompt's first tokens, with a bound
    /// on a worker's load.
    PrefixHash {
        /// The prefix tokens hashed.
        prefix_token_count: usize,
        /// The multiple of the mean load past which a worker is passed over.
        load_factor: f64,
        /// The load over the mean a worker must also exceed to be passed
        /// over.
        balance_abs_threshold: usize,
        /// The token positions a request's head is keyed at.
        cache_boundaries: Vec<usize>,
    },
}

impl PolicySettings {
    /// The policy's name in the configuration, which is its preset's.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::RoundRobin => "round_robin",
            Self::Passthrough => "passthrough",
            Self::CacheAware { .. } => "cache_aware",
            Self::PowerOfTwo { .. } => "power_of_two",
            Self::LeastLoad { .. } => "least_load",
            Self::Bucket { .. } => "bucket",
            Self::Manual { .. } => "manual",
            Self::ConsistentHashing => "consistent_hashing",
            Self::PrefixHash { .. } => "prefix_hash",
        }
    }
}

/// The encode side's policy where the configuration names none: encode
/// placement follows an item's content, not the request's main policy.
static ENCODE_DEFAULT: PolicySettings = PolicySettings::ConsistentHashing;

/// The encode side's profile, by its name in a document of three sides.
const ENCODE_SIDE: &str = "encode";

/// Why a set of routing settings has no policy document to run.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum RoutingSettingsError {
    /// A setting decides a placement and the document has no place for it.
    #[error("{setting} = {value}: {reason}")]
    UnsupportedSetting {
        /// The setting, as the configuration spells its field.
        setting: &'static str,
        /// Its value, as it reads.
        value: String,
        /// Why no document says it.
        reason: &'static str,
    },
    /// A policy's preset holds no stage for one of the policy's settings:
    /// the controller's presets and this module have parted.
    #[error("the {policy} preset has no {stage} stage")]
    NoStage {
        policy: &'static str,
        stage: &'static str,
    },
    /// The same parting over a part that is no stage: a preset holds no
    /// profile, binding or session rule where a setting is written.
    #[error("the {policy} preset has no {part}")]
    NoPart {
        policy: &'static str,
        part: &'static str,
    },
    /// The document's own error: a preset that does not build, or a supplied
    /// document that does not validate.
    #[error(transparent)]
    Document(#[from] SpecError),
    /// The flags change the default document and differ from the supplied one.
    #[error("the routing flags and the policy document disagree at {at}: the flags mean {flags}, the document says {document}")]
    Disagreement {
        /// Where the documents first differ, in their own field names.
        at: String,
        /// The value from the routing flags.
        flags: String,
        /// The value from the supplied document.
        document: String,
    },
}

fn unsupported(
    setting: &'static str,
    value: impl ToString,
    reason: &'static str,
) -> RoutingSettingsError {
    RoutingSettingsError::UnsupportedSetting {
        setting,
        value: value.to_string(),
        reason,
    }
}

impl RoutingSettings {
    /// The policy document these settings say, or the first of them that no
    /// document says.
    ///
    /// A pure function of the settings. The document is not validated
    /// here: the controller validates a document where it is applied, and
    /// names there a stage its build does not run yet.
    pub fn policy_document(&self) -> Result<PolicySpec, RoutingSettingsError> {
        refuse_unsupported(self)?;
        if let Some(both) = both_families(self)? {
            match written(self, both) {
                // A setting the other family's policy reads and no document
                // says leaves that family out. The document then plans the
                // configuration's own family alone.
                Err(RoutingSettingsError::UnsupportedSetting { .. }) => {}
                said => return said,
            }
        }
        written(self, presets(self)?)
    }

    /// The document a gateway of these settings runs. Without a supplied
    /// document, it is the one the routing settings mean. A supplied
    /// document takes effect if the settings describe the same document or
    /// the document of `defaults` for this mode. `defaults` are the
    /// settings of a gateway that sets none: [`RoutingSettings::default`],
    /// or the gateway's own where its configuration defaults a setting to
    /// another value. A conflict names the first difference and the two
    /// values.
    ///
    /// The two overload thresholds are no part of the comparison: no
    /// document carries one, so they stand beside a supplied document
    /// whatever they are, both at none too. What a supplied document's
    /// switch means over no threshold is its host's to say.
    ///
    /// Both launchers refuse a routing setting beside a document before
    /// this runs, so the comparison judges a configuration built by hand.
    /// The generation is no part of it: it numbers the document, and no
    /// routing setting says it. A document that names none is this
    /// gateway's first and takes number 1, since a store applies only a
    /// number above the one it holds.
    pub fn policy_document_in_force(
        &self,
        defaults: &Self,
    ) -> Result<PolicySpec, RoutingSettingsError> {
        let mut said = self.policy_document()?;
        let Some(given) = &self.policy_spec else {
            return Ok(said);
        };
        // The dispatch pins a sticky key by the gateway's own key. A supplied
        // document names its own key sources: where they are not that key,
        // the pin would split a conversation the document keeps together.
        if let Some(pin) = self.grouped_dp_sticky {
            return Err(unsupported(
                "grouped_dp_sticky",
                pin,
                "beside a supplied document the pin would read the gateway's own key, which that document's key sources need not be",
            ));
        }
        let mut given = given.clone();
        if given.generation == 0 {
            given.generation = 1;
        }
        // The default configuration of this gateway's kind: a provider mode
        // has a document under the multi-router setting alone, and a pair
        // mode binds a single request by how the gateway reaches its workers.
        // The two overload thresholds are the gateway's own beside a
        // document, so the default configuration states them as they are.
        let plain = Self {
            mode: self.mode.with_defaults(),
            enable_igw: self.enable_igw,
            connection_mode: self.connection_mode,
            worker_overload_waiting_requests: self.worker_overload_waiting_requests,
            worker_overload_token_usage: self.worker_overload_token_usage,
            ..defaults.clone()
        }
        .policy_document()?;
        if said == plain {
            return Ok(given);
        }
        said.generation = given.generation;
        if said == given {
            return Ok(given);
        }

        let written = |spec| {
            serde_json::to_value(spec)
                .map_err(|error| SpecError::NoCanonicalForm(error.to_string()))
        };
        let flags = written(&said)?;
        let document = written(&given)?;
        if let Some(difference) = first_difference(Some(&flags), Some(&document), String::new()) {
            return Err(difference);
        }

        // A field the reader accepts to report a validation error can be
        // absent from the written form. Let the document name that error.
        given.validate()?;
        Err(RoutingSettingsError::Disagreement {
            at: "document".to_owned(),
            flags: flags.to_string(),
            document: document.to_string(),
        })
    }
}

impl ModeSettings {
    /// This mode with no side policy, so the default document for this mode
    /// carries none of the configured side settings.
    fn with_defaults(&self) -> Self {
        match self {
            Self::Regular => Self::Regular,
            Self::PrefillDecode {
                prefill_policy: _,
                decode_policy: _,
            } => Self::PrefillDecode {
                prefill_policy: None,
                decode_policy: None,
            },
            Self::EncodePrefillDecode {
                encode_policy: _,
                prefill_policy: _,
                decode_policy: _,
            } => Self::EncodePrefillDecode {
                encode_policy: None,
                prefill_policy: None,
                decode_policy: None,
            },
            Self::Provider(provider) => Self::Provider(provider),
        }
    }
}

/// Walk each object in the flags' written order, then through keys that
/// appear only in the supplied document. Walk lists by index.
fn first_difference(
    flags: Option<&Value>,
    document: Option<&Value>,
    at: String,
) -> Option<RoutingSettingsError> {
    match (flags, document) {
        (Some(Value::Object(flags)), Some(Value::Object(document))) => flags
            .keys()
            .chain(document.keys().filter(|key| !flags.contains_key(*key)))
            .find_map(|key| {
                let path = if at.is_empty() {
                    key.clone()
                } else {
                    format!("{at}.{key}")
                };
                first_difference(flags.get(key), document.get(key), path)
            }),
        (Some(Value::Array(flags)), Some(Value::Array(document))) => {
            (0..flags.len().max(document.len())).find_map(|index| {
                first_difference(
                    flags.get(index),
                    document.get(index),
                    format!("{at}[{index}]"),
                )
            })
        }
        _ if flags == document => None,
        _ => {
            let written = |value: Option<&Value>| {
                value.map_or_else(|| "nothing".to_owned(), Value::to_string)
            };
            Some(RoutingSettingsError::Disagreement {
                at,
                flags: written(flags),
                document: written(document),
            })
        }
    }
}

/// The settings that decide a placement today and that no document says.
fn refuse_unsupported(config: &RoutingSettings) -> Result<(), RoutingSettingsError> {
    if config.dp_minimum_tokens_scheduler {
        return Err(unsupported(
            "dp_minimum_tokens_scheduler",
            true,
            "the document has no rank policy: under the controller a rank is a candidate of the profile",
        ));
    }
    Ok(())
}

/// A profile of the document and the policy that decides it.
type Side<'a> = (&'static str, &'a PolicySettings);

/// A document and, binding by binding, the sides its plan runs.
type Planned<'a> = (PolicySpec, Vec<Vec<Side<'a>>>);

/// The presets the mode's policies name, as a document of one binding, with
/// its profiles: one named for the policy under the regular mode, one a side
/// under the pair and the encode modes. A provider mode of a multi-router
/// gateway reads as the regular mode: the document says the policy of its
/// self-hosted workers.
///
/// A pair mode's document binds two families ([`both_families`]), and the
/// encode mode's does in a gateway of many routers. The mode's arm here is
/// then reached only where the main policy's single binding has no
/// document, and plans the mode's own family alone.
fn presets(config: &RoutingSettings) -> Result<Planned<'_>, RoutingSettingsError> {
    let (mode, main) = (&config.mode, &config.policy);
    let (prefill, decode) = (mode.prefill_policy(main), mode.decode_policy(main));
    let regular = || -> Result<_, RoutingSettingsError> {
        Ok((PolicySpec::preset(main.name())?, vec![(main.name(), main)]))
    };
    let (document, sides) = match mode {
        ModeSettings::Regular => regular()?,
        ModeSettings::PrefillDecode {
            prefill_policy: _,
            decode_policy: _,
        } => (
            PolicySpec::pair_presets(prefill.name(), decode.name())?,
            vec![("prefill", prefill), ("decode", decode)],
        ),
        ModeSettings::EncodePrefillDecode {
            encode_policy: _,
            prefill_policy: _,
            decode_policy: _,
        } => {
            let encode = mode.encode_policy(&ENCODE_DEFAULT);
            (
                PolicySpec::epd_presets(encode.name(), prefill.name(), decode.name())?,
                vec![
                    (ENCODE_SIDE, encode),
                    ("prefill", prefill),
                    ("decode", decode),
                ],
            )
        }
        ModeSettings::Provider(_) => match config.provider_alone() {
            Some(provider) => return Err(provider_mode(provider)),
            None => regular()?,
        },
    };
    Ok((document, vec![sides]))
}

/// The document that binds pairs and single requests side by side: the pair
/// binding first, over the prefill and decode phases, and the single binding
/// behind it for every other request. Two configurations say it. A gateway
/// of many routers serves pairs and single workers. A gateway in the pair
/// mode, of many routers or of one, places one single request beside its
/// pairs over HTTP: a token count takes one prefill worker, and the
/// gateway's own routing places it by the main policy. Each family runs
/// under the policies the gateway's own routing gives it. The sides of a
/// pair run under the mode's side policies, which are the main policy in a
/// regular configuration. Single requests run under the main policy.
///
/// Over gRPC or ZMQ the pair mode counts tokens with no worker, as the
/// encode mode does. A gateway of one router then places no single request
/// and its document plans the pair alone: a binding no request reaches
/// would still add the main policy's needs to the document's. A gateway of
/// many routers serves single workers however it reaches them, in the
/// encode mode too: there the first binding is the mode's own, its three
/// sides over their three phases.
///
/// `None` where the document plans one family: a regular configuration of
/// one router, the encode mode of one router, a pair mode of one router
/// over gRPC or ZMQ, and a regular configuration under the manual policy.
/// Its picker leaves the placement to the binding's session rule, and this
/// module does not plan its pairs beside its single requests yet: a pair
/// mode whose two sides are manual is planned, under the one rule. A
/// gateway of one router in the encode mode counts tokens with no worker,
/// so its document binds no single request.
fn both_families(config: &RoutingSettings) -> Result<Option<Planned<'_>>, RoutingSettingsError> {
    let (mode, main) = (&config.mode, &config.policy);
    let (both, encode) = match mode {
        ModeSettings::Regular | ModeSettings::Provider(_) => (
            config.enable_igw && !matches!(main, PolicySettings::Manual { .. }),
            None,
        ),
        ModeSettings::PrefillDecode {
            prefill_policy: _,
            decode_policy: _,
        } => (
            config.enable_igw || config.connection_mode == ConnectionModeSetting::Http,
            None,
        ),
        ModeSettings::EncodePrefillDecode {
            encode_policy: _,
            prefill_policy: _,
            decode_policy: _,
        } => (config.enable_igw, Some(mode.encode_policy(&ENCODE_DEFAULT))),
    };
    if !both {
        return Ok(None);
    }
    let (prefill, decode) = (mode.prefill_policy(main), mode.decode_policy(main));
    let mut sides = vec![("prefill", prefill), ("decode", decode)];
    let mut phases = vec![Phase::Prefill, Phase::Decode];
    let mut document = match encode {
        Some(encode) => {
            sides.insert(0, (ENCODE_SIDE, encode));
            phases.insert(0, Phase::Encode);
            PolicySpec::epd_presets(encode.name(), prefill.name(), decode.name())?
        }
        None => PolicySpec::pair_presets(prefill.name(), decode.name())?,
    };
    for binding in &mut document.bindings {
        binding.matcher.phase.clone_from(&phases);
    }
    let single = PolicySpec::preset(main.name())?;
    document.profiles.extend(single.profiles);
    document.bindings.extend(single.bindings);
    Ok(Some((document, vec![sides, vec![(main.name(), main)]])))
}

/// Writes the configuration's settings over a planned document: each side's
/// policy into its profile, what the sides of a binding share into that
/// binding, and what every side shares into the places the document holds
/// once. The overload block lands on every binding, so one switch and one
/// shed word stand for the pools of each family.
fn written<'a>(
    config: &'a RoutingSettings,
    (mut document, bindings): Planned<'a>,
) -> Result<PolicySpec, RoutingSettingsError> {
    // A plan of several sides runs the pair: the pairing mode is its rule.
    if bindings.iter().any(|sides| sides.len() > 1) {
        document.plans.pd_pair.pairing = config.pd_pairing_mode;
    }
    let mut shared = Shared::default();
    let mut bound = Vec::with_capacity(bindings.len());
    for sides in &bindings {
        for &(side, policy) in sides {
            let profile =
                document
                    .profiles
                    .get_mut(side)
                    .ok_or_else(|| RoutingSettingsError::NoPart {
                        policy: policy.name(),
                        part: "profile",
                    })?;
            write_policy(policy, config, profile, &mut shared)?;
            // The encode side places an item by the item's content: it reads
            // no key, and the override never places it.
            if side != ENCODE_SIDE {
                if let Some(sources) = key_sources(policy, config)? {
                    shared.bound.keys.push((policy.name(), sources));
                }
                shared.bound.overridden |= override_places(policy, config);
            }
        }
        bound.push((sides.len(), std::mem::take(&mut shared.bound)));
    }
    // A pair mode of one router binds its single request for one side
    // route, a token count: that binding leaves the pairs' keys where they
    // were. Many routers serve each family as a fleet of its own, and keep
    // the document's one size for both, as they did. The encode mode of
    // many routers had its own document alone: its rings keep their blocks
    // too, so the single binding costs it no document that was taken.
    let encode_mode = matches!(config.mode, ModeSettings::EncodePrefillDecode { .. });
    if !config.enable_igw || encode_mode {
        rings_keep_their_blocks(shared.block_size, &bound)?;
    }
    shared.write(&mut document)?;
    if document.bindings.is_empty() || document.bindings.len() != bound.len() {
        return Err(RoutingSettingsError::NoPart {
            policy: config.policy.name(),
            part: "binding",
        });
    }
    for (binding, (sides, bound)) in document.bindings.iter_mut().zip(bound) {
        bound.write(config, sides, binding, &mut document.view.headers)?;
    }
    let overload = overload(config);
    for binding in &mut document.bindings {
        binding.overload.clone_from(&overload);
    }
    Ok(document)
}

/// A prefix ring keeps the blocks of its own binding's document, in a
/// gateway of one router and in the encode mode of many routers. The view
/// holds one block size for every profile. A ring names none: its key ends
/// with the last whole block at or under its depth, counted in the view's
/// size. So a size another binding's policy names would end the key at
/// another token, or leave it no whole block, which the controller's check
/// refuses. Such settings are an unsupported setting of the other family,
/// and the document plans the mode's own family alone. Sides of one binding
/// that name a size beside a ring say one document together, as they did.
fn rings_keep_their_blocks(
    block_size: Option<usize>,
    bound: &[(usize, Bound)],
) -> Result<(), RoutingSettingsError> {
    let Some(named) = block_size else {
        return Ok(());
    };
    let silent = usize::try_from(Gates::default().block_size).unwrap_or(usize::MAX);
    let own = |binding: &Bound| binding.block_size.unwrap_or(silent);
    match bound
        .iter()
        .find(|(_, binding)| binding.rings && own(binding) != named)
    {
        Some((_, binding)) => Err(unsupported(
            "block_size",
            format!("{} and {named}", own(binding)),
            "a prefix ring counts its key in the document's one block size, and another binding's policy names another size than the ring's own",
        )),
        None => Ok(()),
    }
}

fn provider_mode(mode: &'static str) -> RoutingSettingsError {
    unsupported(
        "mode",
        mode,
        "a provider mode places its requests by no routing policy",
    )
}

/// Writes one side's policy over its preset: its settings into the stages of
/// its profile, and into the places the document's sides share.
fn write_policy<'a>(
    policy: &'a PolicySettings,
    config: &RoutingSettings,
    profile: &mut ProfileSpec,
    shared: &mut Shared<'a>,
) -> Result<(), RoutingSettingsError> {
    let name = policy.name();
    match policy {
        PolicySettings::Random
        | PolicySettings::RoundRobin
        | PolicySettings::Passthrough
        | PolicySettings::ConsistentHashing => Ok(()),
        PolicySettings::PowerOfTwo { load_metric } => {
            let metric = match *load_metric {
                LoadMetricSetting::ExpectedWait => PowerOfTwoMetric::ExpectedWait,
                LoadMetricSetting::Requests => PowerOfTwoMetric::Requests,
                LoadMetricSetting::LeastRequests => PowerOfTwoMetric::LeastRequests,
                LoadMetricSetting::ResidentTokens { request_tokens } => {
                    PowerOfTwoMetric::ResidentTokens {
                        request_tokens: narrow("load_metric.request_tokens", request_tokens)?,
                    }
                }
            };
            // The preset in its metric's form: the form's scorers and picker
            // under the side's own filters.
            let form = PolicySpec::power_of_two(metric)?
                .profiles
                .remove(name)
                .ok_or(RoutingSettingsError::NoPart {
                    policy: name,
                    part: "profile",
                })?;
            (profile.score, profile.pick) = (form.score, form.pick);
            Ok(())
        }
        PolicySettings::LeastLoad {
            kv_pressure_weight,
            mean_prefill_tokens,
            default_throughput,
        } => {
            let written = LeastLoadParams {
                kv_pressure_weight: unless(*kv_pressure_weight as f32, DEFAULT_KV_PRESSURE_WEIGHT),
                mean_prefill_tokens: unless(*mean_prefill_tokens, DEFAULT_MEAN_PREFILL_TOKENS),
                default_throughput: unless(*default_throughput as f32, DEFAULT_THROUGHPUT),
                ..LeastLoadParams::default()
            };
            if written != LeastLoadParams::default() {
                restage(profile, name, "least-load", written.weight, |weight| {
                    ScoreStage::LeastLoad(LeastLoadParams { weight, ..written })
                })?;
            }
            Ok(())
        }
        PolicySettings::Bucket {
            balance_abs_threshold,
            balance_rel_threshold,
            bucket_adjust_interval_secs,
        } => {
            let written = BucketParams {
                abs_threshold: unless(
                    threshold(*balance_abs_threshold),
                    bucket::DEFAULT_ABS_THRESHOLD,
                ),
                rel_threshold: unless(*balance_rel_threshold, bucket::DEFAULT_REL_THRESHOLD),
                adjust_interval_s: unless(
                    narrow("bucket_adjust_interval_secs", *bucket_adjust_interval_secs)?,
                    bucket::DEFAULT_ADJUST_INTERVAL_S,
                ),
                ..BucketParams::default()
            };
            if written != BucketParams::default() {
                restage(profile, name, "bucket", written.weight, |weight| {
                    ScoreStage::Bucket(BucketParams { weight, ..written })
                })?;
            }
            Ok(())
        }
        PolicySettings::Manual {
            eviction_interval_secs,
            max_idle_secs,
            assignment_mode,
        } => {
            // The manual preset places by the binding's session rule, which
            // every side shares: its settings land there.
            shared.bound.manual_sides += 1;
            share(
                &mut shared.bound.manual,
                (*eviction_interval_secs, *max_idle_secs, *assignment_mode),
                "eviction_interval_secs, max_idle_secs, assignment_mode",
            )
        }
        PolicySettings::PrefixHash {
            prefix_token_count,
            load_factor,
            balance_abs_threshold,
            cache_boundaries,
        } => {
            // The preset names the ring's three numbers itself, so they are
            // written whatever they are.
            let prefix_tokens = Some(narrow("prefix_token_count", *prefix_token_count)?);
            shared.bound.rings = true;
            let bounded_load = Some(BoundedLoadParams {
                factor: Some(*load_factor),
                abs_margin: Some(threshold(*balance_abs_threshold)),
            });
            let bare = ConsistentHashParams::default().weight;
            restage(profile, name, "consistent-hash", bare, |weight| {
                ScoreStage::ConsistentHash(ConsistentHashParams {
                    weight,
                    prefix_tokens,
                    bounded_load,
                })
            })?;
            share(
                &mut shared.boundaries,
                cache_boundaries.as_slice(),
                "cache_boundaries",
            )
        }
        PolicySettings::CacheAware {
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
            cache_boundaries,
            selection_policy,
            selection_accounting_ttl_ms,
        } => {
            if let Some(named) = selection_policy.as_deref() {
                return Err(unsupported(
                    "selection_policy",
                    named,
                    "the document names a picker, and no selection policy but the default",
                ));
            }
            // The gates a silent profile runs under, as the controller builds
            // them.
            let gates = Gates::default();
            let spill = SpillParams {
                abs: unless(threshold(*balance_abs_threshold), gates.abs),
                rel: unless(*balance_rel_threshold, gates.rel),
                prefix_margin_tokens: unless(
                    *balance_prefix_margin_tokens,
                    gates.prefix_margin_tokens,
                ),
                uncached_credit: unless(*balance_uncached_credit, false),
            };
            let affinity = PrefixAffinityParams {
                threshold: unless(*cache_threshold, DEFAULT_THRESHOLD),
                decay: unless(*overlap_decay, 0.0),
                accounting_ttl_ms: unless(*selection_accounting_ttl_ms, 0),
                spill: unless(spill, SpillParams::default()),
                ..PrefixAffinityParams::default()
            };
            if affinity != PrefixAffinityParams::default() {
                restage(
                    profile,
                    name,
                    "prefix-affinity",
                    affinity.weight,
                    |weight| {
                        ScoreStage::PrefixAffinity(PrefixAffinityParams { weight, ..affinity })
                    },
                )?;
            }
            let pressure = KvPressureParams {
                spread: unless(*balance_token_usage_threshold, gates.kv_spread),
                ceiling: unless(*overload_token_usage_threshold, gates.kv_ceiling),
                ..KvPressureParams::default()
            };
            if pressure != KvPressureParams::default() {
                restage(profile, name, "kv-pressure", pressure.weight, |weight| {
                    ScoreStage::KvPressure(KvPressureParams { weight, ..pressure })
                })?;
            }
            let warmup = WarmupSliceParams {
                secs: unless(config.worker_warmup_secs, DEFAULT_WARMUP_SECS),
                share: unless(config.worker_warmup_share, DEFAULT_WARMUP_SHARE),
                blocks: unless(
                    narrow("worker_warmup_blocks", config.worker_warmup_blocks)?,
                    DEFAULT_WARMUP_BLOCKS,
                ),
                thin_ratio: unless(config.worker_warmup_thin_ratio, DEFAULT_WARMUP_THIN_RATIO),
                divert_every: unless(
                    config.worker_warmup_divert_every,
                    DEFAULT_WARMUP_DIVERT_EVERY,
                ),
                // The slice counts a hit as a miss by the affinity's own
                // threshold: the one setting is written in both stages.
                miss_share: unless(*cache_threshold, DEFAULT_THRESHOLD),
                ..WarmupSliceParams::default()
            };
            if warmup != WarmupSliceParams::default() {
                restage(profile, name, "warmup-slice", warmup.weight, |weight| {
                    ScoreStage::WarmupSlice(WarmupSliceParams { weight, ..warmup })
                })?;
            }
            if *selection_temperature != 0.0 {
                profile.pick = PickStage::BestWithTemperature(BestWithTemperatureParams {
                    temperature: *selection_temperature,
                })
                .into();
            }
            share(&mut shared.block_size, *block_size, "block_size")?;
            shared.bound.block_size = Some(*block_size);
            share(
                &mut shared.boundaries,
                cache_boundaries.as_slice(),
                "cache_boundaries",
            )?;
            share(
                &mut shared.index,
                (
                    *cache_index,
                    narrow("cache_ttl_secs", *cache_ttl_secs)?,
                    narrow("max_tree_size", *max_tree_size)?,
                    narrow("eviction_interval_secs", *eviction_interval_secs)?,
                ),
                "cache_index, cache_ttl_secs, max_tree_size, eviction_interval_secs",
            )
        }
    }
}

/// Puts a stage in the place of the profile's stage of that name, at the
/// weight the preset gives it. A stage the preset names bare weighs what its
/// parameters default to in the controller: the caller hands that as `bare`.
fn restage(
    profile: &mut ProfileSpec,
    policy: &'static str,
    stage: &'static str,
    bare: f32,
    written: impl FnOnce(f32) -> ScoreStage,
) -> Result<(), RoutingSettingsError> {
    let missing = RoutingSettingsError::NoStage { policy, stage };
    let Some(entry) = profile.score.iter_mut().find(|entry| entry.name() == stage) else {
        return Err(missing);
    };
    let weight = match entry {
        ScoreSpec::Name(_) => bare,
        ScoreSpec::Params(preset) => preset.weight(),
        ScoreSpec::Added { .. } => return Err(missing),
    };
    *entry = written(weight).into();
    Ok(())
}

/// `value` where it differs from `silent`, what the document means where
/// it says nothing.
fn unless<T: PartialEq>(value: T, silent: T) -> Option<T> {
    (value != silent).then_some(value)
}

/// A number of the configuration in the document's width.
fn narrow<T, U>(setting: &'static str, value: U) -> Result<T, RoutingSettingsError>
where
    T: TryFrom<U>,
    U: Copy + ToString,
{
    T::try_from(value).map_err(|_| unsupported(setting, value, "past the document's range"))
}

/// A balance threshold in the document's width. One past the width lands as
/// the width's largest number: no load reaches either, so the stage that
/// reads it decides the same under both.
fn threshold(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// Seconds of a session rule in the document's width. The rule reads its
/// largest value as for ever, so a setting that large has no place in it.
fn seconds(setting: &'static str, secs: u64) -> Result<u32, RoutingSettingsError> {
    narrow(setting, secs).and_then(|narrowed: u32| {
        unless(narrowed, u32::MAX).ok_or_else(|| {
            unsupported(
                setting,
                secs,
                "the document reads this many seconds as for ever",
            )
        })
    })
}

/// A sticky map's idle time and sweep interval as a session rule's `ttl_s`
/// and `sweep_s`, each with the name of its setting. The map sweeps nothing
/// where either time is zero: a key the table holds stays pinned for ever,
/// which the rule says with its largest number.
fn pin_times(
    (idle_setting, max_idle_secs): (&'static str, u64),
    (sweep_setting, eviction_interval_secs): (&'static str, u64),
) -> Result<(u32, u32), RoutingSettingsError> {
    if max_idle_secs == 0 || eviction_interval_secs == 0 {
        return Ok((u32::MAX, u32::MAX));
    }
    Ok((
        seconds(idle_setting, max_idle_secs)?,
        seconds(sweep_setting, eviction_interval_secs)?,
    ))
}

/// What the sides of a document share. The document holds each once, so the
/// first side to name one sets it, and a side that names another value is
/// refused.
#[derive(Default)]
struct Shared<'a> {
    block_size: Option<usize>,
    boundaries: Option<&'a [usize]>,
    /// The index's kind, entry lifetime, size and sweep interval.
    index: Option<(IndexKind, u32, u64, u32)>,
    /// What the sides of the binding being written share.
    bound: Bound,
}

/// What the sides of one binding share. The binding holds each once.
#[derive(Default)]
struct Bound {
    /// The manual policy's sweep interval, idle time and assignment.
    manual: Option<(u64, u64, Assignment)>,
    manual_sides: usize,
    /// The sticky override places a side's keyed requests
    /// ([`override_places`]).
    overridden: bool,
    /// The sources each keyed side reads a request's key from, under the
    /// side's policy.
    keys: Vec<(&'static str, RoutingKeySpec)>,
    /// The block size a side of the binding names.
    block_size: Option<usize>,
    /// A side of the binding places by a prefix ring, which counts its key
    /// in the view's blocks and names no size ([`rings_keep_their_blocks`]).
    rings: bool,
}

/// A value as the configuration spells it.
fn spelled<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_else(|unspelled| unspelled.to_string())
}

fn share<T: PartialEq + Serialize>(
    held: &mut Option<T>,
    value: T,
    setting: &'static str,
) -> Result<(), RoutingSettingsError> {
    match held {
        Some(held) if *held != value => Err(unsupported(
            setting,
            format!("{} and {}", spelled(held), spelled(&value)),
            "two sides name two values and the document holds one for every side",
        )),
        Some(_) | None => {
            *held = Some(value);
            Ok(())
        }
    }
}

impl Shared<'_> {
    /// Writes what the sides named into the places the document holds once.
    fn write(self, document: &mut PolicySpec) -> Result<(), RoutingSettingsError> {
        if let Some(block_size) = self.block_size {
            document.view.block_size = unless(
                narrow("block_size", block_size)?,
                Gates::default().block_size,
            );
        }
        if let Some(boundaries) = self.boundaries {
            document.view.boundaries = boundaries
                .iter()
                .map(|at| narrow("cache_boundaries", *at))
                .collect::<Result<_, _>>()?;
        }
        let index = self.index.map(|(kind, ttl_s, max_size, sweep_s)| {
            // The gateway's own policy trims its index at its sweep alone,
            // and sweeps nothing at an interval of zero: that index is never
            // trimmed.
            // The document's cadence and size both start at one, so its
            // longest cadence and its largest size say it. No sweep falls
            // inside a process's life, and no write meets the bound.
            let (max_size, sweep_s) = if sweep_s == 0 {
                (u64::MAX, u32::MAX)
            } else {
                (max_size, sweep_s)
            };
            IndexSpec {
                kind,
                ttl_s,
                max_size,
                sweep_s,
            }
        });
        if let Some(index) = index.and_then(|index| unless(index, IndexSpec::default())) {
            document.index.insert("*".to_string(), index);
        }
        Ok(())
    }
}

impl Bound {
    /// Writes what the sides named into their binding, a plan of `sides`
    /// profiles. The headers the manual policy reads are the view's too,
    /// and so are the headers of a key the sticky override places.
    fn write(
        self,
        config: &RoutingSettings,
        sides: usize,
        binding: &mut BindingSpec,
        view_headers: &mut Vec<String>,
    ) -> Result<(), RoutingSettingsError> {
        let Some((eviction_interval_secs, max_idle_secs, assignment_mode)) = self.manual else {
            // A binding names its key's sources once: what its keyed sides
            // read, or nothing, over whatever its presets wrote.
            let key = one_list(self.keys)?.filter(|key| !key.sources.is_empty());
            if let Some(key) = &key {
                within_a_block(key, config)?;
            }
            if self.overridden {
                if sides != 1 {
                    return Err(unsupported(
                        "routing_key_override.enabled",
                        true,
                        "the sticky sessions' rule places single requests alone, and this plan places a pair",
                    ));
                }
                binding.sessions = Some(override_rule(config)?);
                // The rule's cap counts a key's requests on a worker by
                // the key's text, which a header hands where the view
                // lists it.
                *view_headers = key
                    .iter()
                    .flat_map(|key| &key.sources)
                    .filter_map(|source| source.header.clone())
                    .collect();
            }
            binding.routing_key = key;
            return Ok(());
        };
        if self.manual_sides != sides {
            return Err(unsupported(
                "policy",
                "manual",
                "the manual policy places by the binding's one session rule, so every side of a plan is manual or none is",
            ));
        }
        let key = one_list(self.keys)?.ok_or(RoutingSettingsError::NoPart {
            policy: "manual",
            part: "key sources",
        })?;
        let headers: Vec<String> = key
            .sources
            .iter()
            .filter_map(|source| source.header.clone())
            .collect();
        // The rule's own two fields say a list of headers, each at the
        // default length. The request's id and the stripped suffixes take
        // the binding's block, and the rule then names no header.
        let by_rule = !key.strip_lineage
            && key
                .sources
                .iter()
                .all(|source| !source.rid && source.max_bytes.is_none());
        if !by_rule {
            within_a_block(&key, config)?;
        }
        let sticky_header = headers
            .first()
            .filter(|_| by_rule)
            .cloned()
            .unwrap_or_default();
        let (ttl_s, sweep_s) = pin_times(
            ("max_idle_secs", max_idle_secs),
            ("eviction_interval_secs", eviction_interval_secs),
        )?;
        view_headers.clone_from(&headers);
        // A pair of presets names no rule: the manual preset's own is the
        // one the settings are written into. It keeps the preset's word of
        // a request the rule places, the profile's own: the gateway's
        // `manual` policy places a keyed request itself
        // (`model_gateway/src/policies/registry.rs:643`), and no stage
        // counts under its picker.
        let preset = match binding.sessions.take() {
            Some(rule) => Some(rule),
            None => PolicySpec::preset("manual")?
                .bindings
                .pop()
                .and_then(|binding| binding.sessions),
        };
        binding.routing_key = (!by_rule).then_some(key);
        binding.sessions = Some(SessionsSpec {
            sticky_header,
            // One header is the sticky header alone, and the rule lists none.
            headers: if by_rule && headers.len() > 1 {
                headers
            } else {
                Vec::new()
            },
            ttl_s,
            sweep_s,
            assignment: assignment_mode,
            ..preset.ok_or(RoutingSettingsError::NoPart {
                policy: "manual",
                part: "session rule",
            })?
        });
        Ok(())
    }
}

/// The one list of sources the keyed sides read, or the two policies whose
/// lists differ: a binding names one list for every side. The sides are held
/// to the same session sources. Client sources are the hash ring's alone, and
/// no other side's stage reads their key, so the binding takes them from the
/// side that names them, and its target source with them.
fn one_list(
    keys: Vec<(&'static str, RoutingKeySpec)>,
) -> Result<Option<RoutingKeySpec>, RoutingSettingsError> {
    let of_session = |source: &&KeySourceSpec| !source.client && !source.target;
    let mut keys = keys.into_iter();
    let Some((first, mut key)) = keys.next() else {
        return Ok(None);
    };
    for (second, other) in keys {
        let same = other.strip_lineage == key.strip_lineage
            && (other.sources.iter().filter(of_session)).eq(key.sources.iter().filter(of_session));
        if !same {
            return Err(unsupported(
                "policy",
                format!("{first} and {second}"),
                "the two sides read two lists of key sources and a binding names one",
            ));
        }
        if other.sources.len() > key.sources.len() {
            key = other;
        }
    }
    Ok(Some(key))
}

/// A `routing_key` block holds eight sources at most, its client sources
/// and its target source not counted.
fn within_a_block(
    key: &RoutingKeySpec,
    config: &RoutingSettings,
) -> Result<(), RoutingSettingsError> {
    let of_session = |source: &&KeySourceSpec| !source.client && !source.target;
    if key.sources.iter().filter(of_session).count() > MAX_KEY_SOURCES {
        return Err(unsupported(
            "routing_key_override.headers",
            spelled(&config.routing_key_override.headers),
            "a binding names eight sources of its key at most",
        ));
    }
    Ok(())
}

/// The two headers the hash ring reads as a client's key, in its order.
const CLIENT_HEADERS: [&str; 2] = ["x-forwarded-for", "cookie"];

/// The sources a keyed policy reads a request's routing key from, in the
/// order it reads them, as a binding's `routing_key` block names them: an
/// empty list where the policy reads a key and no source is configured.
/// `None` for a policy that reads no key.
///
/// - Every keyed policy reads the configured headers first, lowercase and
///   each once.
/// - `manual` and `consistent_hashing` read the routing key header behind
///   them. Under the override they read the request's own id behind that,
///   and every source loses its turn and retry suffixes.
/// - `prefix_hash` reads the configured headers alone.
/// - `random`, `round_robin`, `power_of_two`, `least_load`, `passthrough`,
///   `bucket` and `cache_aware` read no key themselves. Under the override
///   they read one ([`override_places`]):
///   the configured headers, then the request's own id, each without its
///   turn and retry suffixes. The binding's session rule pins that key
///   ([`override_rule`]).
///
/// Three reads have no document yet and are refused by the setting's name.
/// The gateway's override places a keyed request of `prefix_hash` by its
/// sticky map too, and [`unplaced`] says what it lacks. In a gateway of
/// many routers the override's rule would plan single requests alone,
/// beside pairs the gateway serves too. And under the override the gateway
/// reads the routing key header as sent where no configured header names
/// it, while a block strips every source or none.
///
/// `consistent_hashing` also names its two client headers as client
/// sources: a key from them places a request that carries no session key,
/// and lifts no overload veto. A configured `x-forwarded-for` is named both
/// ways, as the gateway reads it: within the cap as a session's key, and
/// whole as a client's.
///
/// `consistent_hashing` alone names its target header, as its target
/// source ([`target_source`]): a request that
/// carries it goes to the worker its number reaches, and no other policy's
/// request reads it.
fn key_sources(
    policy: &PolicySettings,
    config: &RoutingSettings,
) -> Result<Option<RoutingKeySpec>, RoutingSettingsError> {
    let overridden = config.routing_key_override.enabled;
    let hint = match policy {
        PolicySettings::Manual { .. } | PolicySettings::ConsistentHashing => true,
        PolicySettings::PrefixHash { .. } if !overridden => false,
        _ if override_places(policy, config) => {
            if config.enable_igw {
                return Err(unsupported(
                    "routing_key_override.enabled",
                    true,
                    "the sticky sessions' rule places single requests alone, and a gateway of many routers serves pairs beside them",
                ));
            }
            false
        }
        // Each kind is named: a new one builds once its answer is written.
        PolicySettings::PrefixHash { .. }
        | PolicySettings::Random
        | PolicySettings::RoundRobin
        | PolicySettings::Passthrough
        | PolicySettings::CacheAware { .. }
        | PolicySettings::PowerOfTwo { .. }
        | PolicySettings::LeastLoad { .. }
        | PolicySettings::Bucket { .. } => {
            return if overridden {
                Err(unsupported(
                    "routing_key_override.enabled",
                    true,
                    unplaced(policy),
                ))
            } else {
                Ok(None)
            };
        }
    };
    let header = |name: &str, client| KeySourceSpec {
        header: Some(name.to_string()),
        client,
        ..KeySourceSpec::default()
    };
    let reads = |sources: &[KeySourceSpec], name: &str| {
        sources
            .iter()
            .any(|source| source.header.as_deref() == Some(name))
    };
    let mut sources: Vec<KeySourceSpec> = Vec::new();
    for name in &config.routing_key_override.headers {
        let name = name.to_ascii_lowercase();
        if !reads(&sources, &name) {
            sources.push(header(&name, false));
        }
    }
    if hint && !reads(&sources, MANUAL_SESSION_HEADER) {
        if overridden {
            return Err(unsupported(
                "routing_key_override.headers",
                spelled(&config.routing_key_override.headers),
                "under the override the routing key header is read as sent behind these headers, and a binding strips every source or none",
            ));
        }
        sources.push(header(MANUAL_SESSION_HEADER, false));
    }
    if overridden {
        sources.push(KeySourceSpec {
            rid: true,
            ..KeySourceSpec::default()
        });
    }
    if matches!(policy, PolicySettings::ConsistentHashing) {
        sources.extend(CLIENT_HEADERS.map(|name| header(name, true)));
        sources.push(target_source());
    }
    Ok(Some(RoutingKeySpec {
        sources,
        strip_lineage: overridden,
    }))
}

/// Whether the binding's session rule places a keyed request of `policy`
/// ([`override_rule`]): the sticky override is on, and the policy's own
/// stages place a single request by nothing its key hands them. The
/// gateway's sticky map places a keyed request of every policy but
/// `manual` and `consistent_hashing`, which place by the key themselves.
/// `prefix_hash` alone is left out here and refused, for what its answer
/// still lacks ([`unplaced`]).
///
/// - `random` draws.
/// - `round_robin` takes turns. The gateway asks one policy object for a
///   request with no key, for a key's first request and for a key at its
///   pin's cap (`model_gateway/src/policies/registry.rs:664`, `:774`,
///   `:745`), so the three take the turns of one rotation, and a request
///   that follows its pin takes none (`:736-739`). The rotation is the
///   list's own, by row, and where the row's worker takes no request, the
///   rotation of the workers that do
///   (`model_gateway/src/policies/round_robin.rs:142-168`). This crate's
///   stage turns the same two for the three.
/// - `power_of_two` and `least_load` compare workers by load. Both score
///   a worker by its report and by the requests the policy placed on it
///   since. A request the rule placed is none of those: the rule keeps it
///   from the profile ([`override_rule`]), so in a gateway's own process
///   no stage hears its dispatch (`lease_events::stages_hear`), as the
///   gateway's policy hears of no request its sticky map placed
///   (`model_gateway/src/policies/registry.rs:736-739`). Both read such a
///   request in the count of requests the gateway holds on the worker. A
///   controller keeps the rule's word on the lease it files, and tells no
///   stage that dispatch either.
/// - `passthrough` sends every request to the first worker that takes one.
/// - `bucket` draws for a single request, and the rule plans single
///   requests alone. Its stage places a prefill decision by the sizes it
///   booked and draws for every other. The gateway's policy holds a bucket
///   of sizes for the prefill workers of a pair alone, and draws among
///   single workers (`model_gateway/src/policies/registry.rs:1564-1575`,
///   `model_gateway/src/policies/bucket.rs:186-191`).
/// - `cache_aware` reads who was sent a prompt. The gateway's policy writes
///   a prompt where it placed the request, and it is not asked behind a pin
///   its sticky map followed nor for a key the map's own assignment placed
///   (`model_gateway/src/policies/registry.rs:736-739`, `:756`, `:782`).
///   In a gateway's own process the record of sent prefixes, or a token
///   tree the host lends, takes a prompt behind a request the profile's
///   stages place, as under the policy alone, and behind none the rule
///   places alone: the rule keeps those from the profile
///   ([`override_rule`]), and the record's writes read that word
///   (`lease_events::stages_hear`). A controller's decision is this
///   engine's, so its record follows the word too.
fn override_places(policy: &PolicySettings, config: &RoutingSettings) -> bool {
    let placed = matches!(
        policy,
        PolicySettings::Random
            | PolicySettings::RoundRobin
            | PolicySettings::PowerOfTwo { .. }
            | PolicySettings::LeastLoad { .. }
            | PolicySettings::Passthrough
            | PolicySettings::Bucket { .. }
            | PolicySettings::CacheAware { .. }
    );
    config.routing_key_override.enabled && placed
}

/// Why the sticky override has no document yet under a policy whose keyed
/// requests the gateway's sticky map places. `prefix_hash` is the one: it
/// hashes a key a header handed and no key the request's own id handed,
/// and the gateway places that request by its prompt. A binding has one
/// key, and its ring places by it whatever its source.
///
/// The kinds the rule or their own key places never reach this function:
/// each is named, so a new kind builds once its sentence is written.
fn unplaced(policy: &PolicySettings) -> &'static str {
    match policy {
        PolicySettings::PrefixHash { .. } => {
            "under the override this policy's ring would place a key the request's own id handed, and the gateway places that request by its prompt"
        }
        PolicySettings::Manual { .. }
        | PolicySettings::ConsistentHashing
        | PolicySettings::Random
        | PolicySettings::RoundRobin
        | PolicySettings::Passthrough
        | PolicySettings::CacheAware { .. }
        | PolicySettings::PowerOfTwo { .. }
        | PolicySettings::LeastLoad { .. }
        | PolicySettings::Bucket { .. } => {
            "the sticky sessions' rule has no document under this policy yet"
        }
    }
}

/// The requests of one key a worker carries under the sticky override
/// before the key's next request is placed again: the number of the
/// gateway's own sticky map.
const OVERRIDE_INFLIGHT_CAP: u16 = 2;

/// The session rule the sticky override places a keyed request by, as the
/// gateway's own sticky map keeps a key:
///
/// - A key's first request is placed by the override's assignment: the
///   policy's own stages under `delegate`, its default.
/// - A worker carries two requests of one key. The next is placed again,
///   and the key's pin moves only to another worker under that count.
/// - Where a key's worker stops taking requests, the worker placed next
///   leads the key's pins under `delegate`. Under the map's own
///   assignments it joins behind the first, which leads again once it
///   takes requests.
/// - An idle key is forgotten after the override's idle time, at its
///   sweep, and never where either time is zero ([`pin_times`]): the
///   override keeps its keys in the manual policy's own map.
/// - A request the rule places alone is kept from the profile
///   ([`RulePlaced::Untold`]): the gateway's policy hears of no request
///   its sticky map placed
///   (`model_gateway/src/policies/registry.rs:736-739`). No other rule
///   these settings write says so.
///
/// The rule names no header: the binding's `routing_key` block lists the
/// key's sources ([`key_sources`]).
fn override_rule(config: &RoutingSettings) -> Result<SessionsSpec, RoutingSettingsError> {
    let sticky = &config.routing_key_override;
    let (ttl_s, sweep_s) = pin_times(
        ("routing_key_override.max_idle_secs", sticky.max_idle_secs),
        (
            "routing_key_override.eviction_interval_secs",
            sticky.eviction_interval_secs,
        ),
    )?;
    Ok(SessionsSpec {
        sticky_header: String::new(),
        headers: Vec::new(),
        ttl_s,
        sweep_s,
        assignment: sticky.assignment_mode,
        inflight_cap: OVERRIDE_INFLIGHT_CAP,
        issue_session_id: false,
        recovery: match sticky.assignment_mode {
            Assignment::Delegate => Recovery::Latest,
            Assignment::Random | Assignment::MinLoad | Assignment::MinGroup => Recovery::Original,
        },
        rule_placed: RulePlaced::Untold,
    })
}

/// The binding's overload block, or none where the configuration stands at
/// the block's own defaults: its switch and its shed word. The two
/// thresholds travel in no document: they are the default a gateway builds
/// its worker objects with, and a worker's registration states its own over
/// them.
fn overload(config: &RoutingSettings) -> Option<OverloadSpec> {
    // A threshold at none switches its signal off in the gateway, on its
    // worker objects. With the veto on, one signal off leaves the block on,
    // and both off is the block off.
    let waiting = config.worker_overload_waiting_requests;
    let usage = config.worker_overload_token_usage;
    let written = OverloadSpec {
        enabled: config.worker_overload_protection && (waiting.is_some() || usage.is_some()),
        shed_when_all_vetoed: config.worker_overload_shed,
        ..OverloadSpec::default()
    };
    unless(written, OverloadSpec::default())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        document::spec::{AppliedSpec, PRESETS},
        engine::framework::{InTheWay, Place},
    };

    /// The settings of a gateway that sets none, as its configuration
    /// defaults them. The document they mean is the `random` preset.
    fn defaults() -> RoutingSettings {
        RoutingSettings::default()
    }

    /// The cache-aware policy with these five settings, and every other at
    /// the value a gateway's configuration file gets where it leaves the
    /// setting out.
    fn cache_aware(
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        eviction_interval_secs: u64,
        max_tree_size: usize,
    ) -> PolicySettings {
        PolicySettings::CacheAware {
            cache_threshold,
            balance_abs_threshold,
            balance_rel_threshold,
            balance_prefix_margin_tokens: 0,
            balance_uncached_credit: false,
            eviction_interval_secs,
            max_tree_size,
            block_size: 16,
            balance_token_usage_threshold: 1.0,
            overload_token_usage_threshold: 1.0,
            overlap_decay: 0.0,
            selection_temperature: 0.0,
            cache_index: IndexKind::Tree,
            cache_ttl_secs: 180,
            cache_boundaries: Vec::new(),
            selection_policy: None,
            selection_accounting_ttl_ms: 0,
        }
    }

    /// The check a gateway passes a document through before it runs it, with
    /// the header a gateway keeps to itself denied to the view.
    fn validated(document: &PolicySpec) -> Result<AppliedSpec, SpecError> {
        AppliedSpec::validate_with(document, &["x-smg-tenant-id"])
    }

    fn config(policy: PolicySettings) -> RoutingSettings {
        RoutingSettings {
            policy,
            ..defaults()
        }
    }

    /// The policy of a preset's name with every setting at its flag's
    /// default.
    fn at_flag_defaults(name: &str) -> PolicySettings {
        match name {
            "random" => PolicySettings::Random,
            "round_robin" => PolicySettings::RoundRobin,
            "passthrough" => PolicySettings::Passthrough,
            "consistent_hashing" => PolicySettings::ConsistentHashing,
            "cache_aware" => cache_aware(0.3, 64, 1.5, 120, 67_108_864),
            "power_of_two" => PolicySettings::PowerOfTwo {
                load_metric: LoadMetricSetting::ExpectedWait,
            },
            "least_load" => PolicySettings::LeastLoad {
                kv_pressure_weight: 0.15,
                mean_prefill_tokens: 1024,
                default_throughput: 2000.0,
            },
            "bucket" => PolicySettings::Bucket {
                balance_abs_threshold: 64,
                balance_rel_threshold: 1.5,
                bucket_adjust_interval_secs: 5,
            },
            "manual" => PolicySettings::Manual {
                eviction_interval_secs: 60,
                max_idle_secs: 14_400,
                assignment_mode: Assignment::Random,
            },
            "prefix_hash" => PolicySettings::PrefixHash {
                prefix_token_count: 256,
                load_factor: 1.25,
                balance_abs_threshold: 10,
                cache_boundaries: Vec::new(),
            },
            other => {
                panic!("the controller has a preset this module converts no policy to: {other}")
            }
        }
    }

    fn preset(name: &str) -> PolicySpec {
        PolicySpec::preset(name).expect("a preset of this build")
    }

    #[test]
    fn with_no_document_the_flags_document_is_in_force() {
        let config = config(PolicySettings::RoundRobin);
        assert_eq!(
            config.policy_document_in_force(&defaults()),
            Ok(preset("round_robin"))
        );
    }

    #[test]
    fn a_given_document_wins_where_no_routing_setting_is_set() {
        for (mode, document) in [
            (ModeSettings::Regular, preset("round_robin")),
            (
                ModeSettings::PrefillDecode {
                    prefill_policy: None,
                    decode_policy: None,
                },
                PolicySpec::pair_presets("round_robin", "random").expect("paired presets"),
            ),
        ] {
            let config = RoutingSettings {
                mode,
                policy_spec: Some(document.clone()),
                ..defaults()
            };
            assert_eq!(config.policy_document_in_force(&defaults()), Ok(document));
        }
    }

    #[test]
    fn a_given_document_that_the_flags_mean_too_is_in_force() {
        let document = preset("round_robin");
        let config = RoutingSettings {
            policy_spec: Some(document.clone()),
            ..config(PolicySettings::RoundRobin)
        };
        assert_eq!(config.policy_document_in_force(&defaults()), Ok(document));
    }

    #[test]
    fn the_generation_numbers_the_document_and_is_no_part_of_the_comparison() {
        for (written, runs) in [(0, 1), (1, 1), (7, 7)] {
            let mut document = preset("round_robin");
            document.generation = written;
            let mut in_force = document.clone();
            in_force.generation = runs;
            for settings in [defaults(), config(PolicySettings::RoundRobin)] {
                let config = RoutingSettings {
                    policy_spec: Some(document.clone()),
                    ..settings
                };
                assert_eq!(
                    config.policy_document_in_force(&defaults()).as_ref(),
                    Ok(&in_force),
                    "generation {written}"
                );
            }
        }
    }

    #[test]
    fn a_routing_flag_that_says_something_else_is_refused_with_both_values() {
        let mut document = preset("bucket");
        let profile = document.profiles.remove("bucket").expect("the profile");
        document.profiles.insert("power_of_two".to_owned(), profile);
        document.bindings[0].profile = Some("power_of_two".to_owned());
        let config = RoutingSettings {
            policy_spec: Some(document),
            ..config(at_flag_defaults("power_of_two"))
        };
        let refusal = config
            .policy_document_in_force(&defaults())
            .expect_err("different stages");
        assert_eq!(
            refusal.to_string(),
            r#"the routing flags and the policy document disagree at profiles.power_of_two.score[0]: the flags mean "least-load", the document says "bucket""#
        );
    }

    #[test]
    fn a_side_policy_that_says_something_else_is_refused_with_both_values() {
        let config = RoutingSettings {
            mode: ModeSettings::PrefillDecode {
                prefill_policy: Some(PolicySettings::RoundRobin),
                decode_policy: None,
            },
            policy_spec: Some(
                PolicySpec::pair_presets("random", "random").expect("paired presets"),
            ),
            ..defaults()
        };
        assert_eq!(
            config.policy_document_in_force(&defaults()),
            Err(RoutingSettingsError::Disagreement {
                at: "profiles.prefill.pick".to_owned(),
                flags: r#""round-robin""#.to_owned(),
                document: r#""random""#.to_owned(),
            }),
        );
    }

    #[test]
    fn a_part_one_side_lacks_reads_nothing() {
        let document = edited(preset("round_robin"), |form| {
            form["bindings"][0]["shadow"] = json!("round_robin");
        });
        let mut config = RoutingSettings {
            policy_spec: Some(document),
            ..config(PolicySettings::RoundRobin)
        };
        assert_eq!(
            config.policy_document_in_force(&defaults()),
            Err(RoutingSettingsError::Disagreement {
                at: "bindings[0].shadow".to_owned(),
                flags: "nothing".to_owned(),
                document: r#""round_robin""#.to_owned(),
            }),
        );

        let mut document = preset("round_robin");
        let binding = serde_json::to_string(&document.bindings[0]).expect("the binding serializes");
        document.bindings.clear();
        config.policy_spec = Some(document);
        assert_eq!(
            config.policy_document_in_force(&defaults()),
            Err(RoutingSettingsError::Disagreement {
                at: "bindings[0]".to_owned(),
                flags: binding,
                document: "nothing".to_owned(),
            }),
        );
    }

    #[test]
    fn a_kind_in_an_otherwise_equal_document_is_refused_by_name() {
        let document = edited(preset("round_robin"), |form| {
            form["kind"] = json!("PolicySpec");
        });
        let config = RoutingSettings {
            policy_spec: Some(document),
            ..config(PolicySettings::RoundRobin)
        };
        assert_eq!(
            config.policy_document_in_force(&defaults()),
            Err(RoutingSettingsError::Document(SpecError::Kind(
                "PolicySpec".to_owned()
            )))
        );
    }

    /// A document with places of its JSON form written over.
    fn edited(document: PolicySpec, edit: impl FnOnce(&mut Value)) -> PolicySpec {
        let mut form = serde_json::to_value(document).expect("a document serializes");
        edit(&mut form);
        serde_json::from_value(form).expect("the edited document")
    }

    fn refusal(config: &RoutingSettings) -> (&'static str, String) {
        match config.policy_document() {
            Err(RoutingSettingsError::UnsupportedSetting { setting, value, .. }) => {
                (setting, value)
            }
            other => panic!("expected a setting refused by name, got {other:?}"),
        }
    }

    /// The settings a configuration leaves out are those of a gateway that
    /// sets none: the `random` preset, with nothing written over it.
    #[test]
    fn the_settings_of_a_gateway_that_sets_none_mean_the_random_preset() {
        assert_eq!(
            RoutingSettings::default().policy_document(),
            Ok(preset("random"))
        );
    }

    #[test]
    fn each_policy_name_at_its_flags_defaults_is_its_preset() {
        for name in PRESETS {
            let policy = at_flag_defaults(name);
            assert_eq!(
                policy.name(),
                name,
                "the policy and the preset share one name"
            );
            assert_eq!(
                config(policy).policy_document(),
                Ok(preset(name)),
                "{name} at its flags' defaults is the preset, with nothing written over it"
            );
        }
    }

    /// A threshold at none switches its signal off in the gateway, on its
    /// worker objects, and no document carries a threshold. With the veto
    /// on, one signal off leaves the block switched on, whatever the other
    /// is: the settings mean the preset, with nothing written over it. Both
    /// off is the block switched off, and the shed word stays where it is
    /// written.
    #[test]
    fn one_overload_signal_off_leaves_the_block_on_and_both_off_switch_it_off() {
        let thresholds = |waiting: Option<usize>, usage: Option<f64>| {
            let mut config = config(at_flag_defaults("round_robin"));
            config.worker_overload_waiting_requests = waiting;
            config.worker_overload_token_usage = usage;
            config
        };
        for (waiting, usage) in [
            (None, Some(0.8)),
            (None, Some(0.5)),
            (Some(8), None),
            (Some(32), None),
        ] {
            assert_eq!(
                thresholds(waiting, usage).policy_document(),
                Ok(preset("round_robin")),
                "waiting {waiting:?}, usage {usage:?}"
            );
        }
        let switched_off = edited(preset("round_robin"), |document| {
            document["bindings"][0]["overload"] = json!({ "enabled": false });
        });
        assert_eq!(
            thresholds(None, None).policy_document(),
            Ok(switched_off.clone())
        );
        // With the protection off the thresholds decide nothing: the block
        // is off whatever they are.
        let mut unprotected = thresholds(None, Some(0.8));
        unprotected.worker_overload_protection = false;
        assert_eq!(unprotected.policy_document(), Ok(switched_off));
        // The shed word does not switch the block on.
        let mut shedding = thresholds(None, None);
        shedding.worker_overload_shed = true;
        assert_eq!(
            shedding.policy_document(),
            Ok(edited(preset("round_robin"), |document| {
                document["bindings"][0]["overload"] =
                    json!({ "enabled": false, "shed_when_all_vetoed": true });
            }))
        );
    }

    /// The converter puts a written stage in the place of a preset's stage
    /// at the preset's weight, and carries no other parameter of it. So a
    /// preset's scorer that has parameters has a weight alone. A preset that
    /// gains another parameter fails here first, and the converter carries
    /// that parameter from then on. `consistent-hash` is the one stage the
    /// converter writes whole from the configuration.
    #[test]
    fn every_presets_scorer_with_parameters_has_a_weight_alone() {
        for name in PRESETS {
            let form = serde_json::to_value(preset(name)).expect("a document serializes");
            let profiles = form["profiles"].as_object().expect("the profiles");
            for (profile, stages) in profiles {
                for scorer in stages["score"].as_array().into_iter().flatten() {
                    let Some(with_parameters) = scorer.as_object() else {
                        continue;
                    };
                    for (stage, parameters) in with_parameters {
                        if stage == "consistent-hash" {
                            continue;
                        }
                        let named: Vec<&String> = parameters
                            .as_object()
                            .into_iter()
                            .flatten()
                            .map(|(name, _)| name)
                            .collect();
                        assert_eq!(named, ["weight"], "{name}: {profile}: {stage}");
                    }
                }
            }
        }
    }

    #[test]
    fn power_of_two_lands_on_the_form_of_its_metric() {
        for (load_metric, form) in [
            (
                LoadMetricSetting::ExpectedWait,
                PowerOfTwoMetric::ExpectedWait,
            ),
            (LoadMetricSetting::Requests, PowerOfTwoMetric::Requests),
            (
                LoadMetricSetting::LeastRequests,
                PowerOfTwoMetric::LeastRequests,
            ),
            (
                LoadMetricSetting::ResidentTokens {
                    request_tokens: 512,
                },
                PowerOfTwoMetric::ResidentTokens {
                    request_tokens: 512,
                },
            ),
        ] {
            assert_eq!(
                config(PolicySettings::PowerOfTwo { load_metric }).policy_document(),
                Ok(PolicySpec::power_of_two(form).expect("a form of the preset")),
                "{load_metric:?} is the preset in that metric's form"
            );
        }
    }

    #[test]
    fn a_pair_mode_is_the_pair_of_presets_under_its_pairing_rule() {
        let pair = RoutingSettings {
            mode: ModeSettings::PrefillDecode {
                prefill_policy: Some(PolicySettings::RoundRobin),
                decode_policy: None,
            },
            pd_pairing_mode: Pairing::Strict,
            ..config(at_flag_defaults("least_load"))
        };
        let mut expected =
            PolicySpec::pair_presets("round_robin", "least_load").expect("a pair of presets");
        expected.plans.pd_pair.pairing = Pairing::Strict;
        assert_eq!(
            pair.policy_document(),
            Ok(with_single(expected, preset("least_load"))),
            "a side with no policy of its own takes the main one, and the pairing mode is the plan's"
        );
    }

    #[test]
    fn an_encode_side_with_no_policy_is_placed_by_consistent_hashing() {
        let encode_pair = RoutingSettings {
            mode: ModeSettings::EncodePrefillDecode {
                encode_policy: None,
                prefill_policy: None,
                decode_policy: Some(PolicySettings::Random),
            },
            ..config(PolicySettings::RoundRobin)
        };
        assert_eq!(
            encode_pair.policy_document(),
            Ok(
                PolicySpec::epd_presets("consistent_hashing", "round_robin", "random")
                    .expect("the encode pair's presets")
            ),
        );
    }

    #[test]
    fn a_cache_aware_setting_lands_in_the_stage_that_reads_it() {
        let tuned = RoutingSettings {
            policy: PolicySettings::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                balance_prefix_margin_tokens: 1024,
                balance_uncached_credit: true,
                eviction_interval_secs: 60,
                max_tree_size: 16_777_216,
                block_size: 256,
                balance_token_usage_threshold: 0.3,
                overload_token_usage_threshold: 0.9,
                overlap_decay: 0.5,
                selection_temperature: 0.05,
                cache_index: IndexKind::Hash,
                cache_ttl_secs: 300,
                cache_boundaries: vec![512, 2048],
                selection_policy: None,
                selection_accounting_ttl_ms: 200,
            },
            worker_warmup_secs: 0,
            worker_warmup_blocks: 2048,
            ..defaults()
        };
        let expected = edited(preset("cache_aware"), |document| {
            document["view"] = json!({ "block_size": 256, "boundaries": [512, 2048] });
            document["index"] = json!({
                "*": { "kind": "hash", "ttl_s": 300, "max_size": 16_777_216, "sweep_s": 60 }
            });
            let profile = &mut document["profiles"]["cache_aware"];
            profile["score"][0] = json!({
                "warmup-slice": { "weight": 8.0, "secs": 0, "blocks": 2048, "miss_share": 0.5 }
            });
            profile["score"][1] = json!({ "prefix-affinity": {
                "weight": 4.0,
                "threshold": 0.5,
                "decay": 0.5,
                "accounting_ttl_ms": 200,
                "spill": {
                    "abs": 32, "rel": 1.1, "prefix_margin_tokens": 1024, "uncached_credit": true
                },
            }});
            profile["score"][2] = json!({ "kv-pressure": { "spread": 0.3, "ceiling": 0.9 } });
            profile["pick"] = json!({ "best-with-temperature": { "temperature": 0.05 } });
        });
        assert_eq!(tuned.policy_document(), Ok(expected));
    }

    #[test]
    fn the_settings_of_the_other_policies_land_in_their_stages() {
        let least_load = config(PolicySettings::LeastLoad {
            kv_pressure_weight: 0.3,
            mean_prefill_tokens: 1024,
            default_throughput: 2000.0,
        });
        assert_eq!(
            least_load.policy_document(),
            Ok(edited(preset("least_load"), |document| {
                document["profiles"]["least_load"]["score"][0] =
                    json!({ "least-load": { "kv_pressure_weight": 0.3 } });
            })),
            "one setting off its default is the one parameter written"
        );
        let bucket = config(PolicySettings::Bucket {
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.5,
            bucket_adjust_interval_secs: 10,
        });
        assert_eq!(
            bucket.policy_document(),
            Ok(edited(preset("bucket"), |document| {
                document["profiles"]["bucket"]["score"][0] =
                    json!({ "bucket": { "abs_threshold": 32, "adjust_interval_s": 10 } });
            })),
        );
        let prefix_hash = config(PolicySettings::PrefixHash {
            prefix_token_count: 512,
            load_factor: 1.5,
            balance_abs_threshold: 4,
            cache_boundaries: vec![256, 1024, 4096],
        });
        assert_eq!(
            prefix_hash.policy_document(),
            Ok(edited(preset("prefix_hash"), |document| {
                document["view"] = json!({ "boundaries": [256, 1024, 4096] });
                document["profiles"]["prefix_hash"]["score"][0] = json!({ "consistent-hash": {
                    "weight": 8.0,
                    "prefix_tokens": 512,
                    "bounded_load": { "factor": 1.5, "abs_margin": 4 },
                }});
            })),
        );
    }

    #[test]
    fn the_manual_policy_writes_the_bindings_session_rule() {
        let manual = PolicySettings::Manual {
            eviction_interval_secs: 30,
            max_idle_secs: 3600,
            assignment_mode: Assignment::MinGroup,
        };
        let rule = |document: &mut Value| {
            let sessions = &mut document["bindings"][0]["sessions"];
            sessions["ttl_s"] = json!(3600);
            sessions["sweep_s"] = json!(30);
            sessions["assignment"] = json!("min_group");
        };
        assert_eq!(
            config(manual.clone()).policy_document(),
            Ok(edited(preset("manual"), rule))
        );

        // Either time at zero sweeps nothing: a key stays pinned for ever.
        let pinned = config(PolicySettings::Manual {
            eviction_interval_secs: 30,
            max_idle_secs: 0,
            assignment_mode: Assignment::Random,
        })
        .policy_document()
        .expect("a document");
        let for_ever = |document: &mut Value| {
            let sessions = &mut document["bindings"][0]["sessions"];
            sessions["ttl_s"] = json!(u32::MAX);
            sessions["sweep_s"] = json!(u32::MAX);
        };
        assert_eq!(pinned, edited(preset("manual"), for_ever));
        assert!(pinned.validate().is_ok(), "the controller takes the rule");

        // The sweep at zero beside an idle time: the gateway starts no
        // sweep, so the key stays pinned for ever too.
        let unswept = config(PolicySettings::Manual {
            eviction_interval_secs: 0,
            max_idle_secs: 3600,
            assignment_mode: Assignment::Random,
        })
        .policy_document()
        .expect("a document");
        assert_eq!(unswept, edited(preset("manual"), for_ever));

        // A pair of manual sides: the pair of presets names no rule, and the
        // manual preset's own is the one written. The single binding behind
        // the pair is the regular document's, under the same rule.
        let mut pair = PolicySpec::pair_presets("manual", "manual").expect("a pair of presets");
        pair.bindings[0].sessions = preset("manual").bindings[0].sessions.clone();
        let both = RoutingSettings {
            mode: ModeSettings::PrefillDecode {
                prefill_policy: None,
                decode_policy: None,
            },
            ..config(manual.clone())
        };
        assert_eq!(
            both.policy_document(),
            Ok(with_single(
                edited(pair, rule),
                edited(preset("manual"), rule)
            ))
        );

        let one_side = RoutingSettings {
            mode: ModeSettings::PrefillDecode {
                prefill_policy: None,
                decode_policy: Some(PolicySettings::RoundRobin),
            },
            ..config(manual)
        };
        assert_eq!(
            refusal(&one_side),
            ("policy", "manual".to_string()),
            "one session rule serves the whole binding, so a manual side beside another policy is refused"
        );
    }

    #[test]
    fn the_headers_of_a_routing_key_land_where_the_policy_reads_its_key() {
        let keyed = |policy: &str, headers: &[&str]| {
            let mut keyed = config(at_flag_defaults(policy));
            keyed.routing_key_override.headers =
                headers.iter().map(|name| (*name).to_string()).collect();
            keyed
        };

        // The manual policy reads the configured headers, then the routing
        // key header. Its rule names them: the first is the sticky one, and
        // the rule lists them where there is more than one, each once and
        // lowercase.
        let both = |document: &mut Value| {
            document["view"]["headers"] = json!(["x-session-id", "x-smg-routing-key"]);
            let sessions = &mut document["bindings"][0]["sessions"];
            sessions["sticky_header"] = json!("x-session-id");
            sessions["headers"] = json!(["x-session-id", "x-smg-routing-key"]);
        };
        let listed = ["X-Session-Id", "x-smg-routing-key", "x-session-id"];
        assert_eq!(
            keyed("manual", &listed).policy_document(),
            Ok(edited(preset("manual"), both))
        );
        assert_eq!(
            keyed("manual", &["x-session-id"]).policy_document(),
            Ok(edited(preset("manual"), both)),
            "the routing key header is read behind a configured one that is not it"
        );
        assert_eq!(
            keyed("manual", &[]).policy_document(),
            Ok(preset("manual")),
            "with no header configured the manual policy still reads the routing key header"
        );

        // The prefix ring reads the configured headers alone, and none is no
        // block. The hash ring reads the routing key header behind them.
        let sources = |name: &str, sources: Value| {
            edited(preset(name), |document| {
                document["bindings"][0]["routing_key"] = json!({ "sources": sources });
            })
        };
        assert_eq!(
            keyed("prefix_hash", &["x-session-id", "X-Other-Key"]).policy_document(),
            Ok(sources(
                "prefix_hash",
                json!([{ "header": "x-session-id" }, { "header": "x-other-key" }])
            ))
        );
        assert_eq!(
            keyed("prefix_hash", &[]).policy_document(),
            Ok(edited(preset("prefix_hash"), |document| {
                document["bindings"][0]["routing_key"] = Value::Null;
            }))
        );
        assert_eq!(
            keyed("consistent_hashing", &["x-session-id"]).policy_document(),
            Ok(sources(
                "consistent_hashing",
                json!([
                    { "header": "x-session-id" },
                    { "header": "x-smg-routing-key" },
                    { "header": "x-forwarded-for", "client": true },
                    { "header": "cookie", "client": true },
                    { "header": "x-smg-target-worker", "target": true }
                ])
            ))
        );

        // Eight sources bound a block. The manual policy's rule holds the
        // configured headers, bounded by the view: 256 with the routing key
        // header.
        let eight: Vec<String> = (0..8).map(|n| format!("x-key-{n}")).collect();
        let eight: Vec<&str> = eight.iter().map(String::as_str).collect();
        assert_eq!(
            refusal(&keyed("consistent_hashing", &eight)).0,
            "routing_key_override.headers"
        );
        assert!(keyed("manual", &eight).policy_document().is_ok());
        // The hash ring's two client sources are counted apart: seven
        // configured headers and the routing key header are its eight.
        let seven = keyed("consistent_hashing", &eight[..7]);
        let written = seven.policy_document().expect("eight sources");
        assert!(validated(&written).is_ok());

        // The hash ring reads a configured `x-forwarded-for` both ways,
        // within the cap as a session's key and whole as a client's. The
        // document names it twice, and the controller's check takes it.
        let both_ways = keyed("consistent_hashing", &["x-forwarded-for"]);
        let document = sources(
            "consistent_hashing",
            json!([
                { "header": "x-forwarded-for" },
                { "header": "x-smg-routing-key" },
                { "header": "x-forwarded-for", "client": true },
                { "header": "cookie", "client": true },
                { "header": "x-smg-target-worker", "target": true }
            ]),
        );
        assert_eq!(both_ways.policy_document().as_ref(), Ok(&document));
        assert!(validated(&document).is_ok());

        // No other policy reads a key, so its document names no source.
        let keyed_policies = ["manual", "prefix_hash", "consistent_hashing"];
        for name in PRESETS
            .iter()
            .filter(|name| !keyed_policies.contains(*name))
        {
            assert_eq!(
                keyed(name, &["x-session-id"]).policy_document(),
                Ok(preset(name)),
                "{name}"
            );
        }
    }

    #[test]
    fn the_override_adds_the_requests_id_where_a_policy_keeps_its_own_key() {
        let overridden = |policy: &str| {
            let mut overridden = config(at_flag_defaults(policy));
            overridden.routing_key_override.enabled = true;
            overridden
        };

        // The hash ring keeps its own placement under the override. Its
        // sources gain the request's id behind the headers, and every source
        // loses its turn and retry suffixes.
        assert_eq!(
            overridden("consistent_hashing").policy_document(),
            Ok(edited(preset("consistent_hashing"), |document| {
                document["bindings"][0]["routing_key"] = json!({
                    "sources": [
                        { "header": "x-smg-routing-key" },
                        { "rid": true },
                        { "header": "x-forwarded-for", "client": true },
                        { "header": "cookie", "client": true },
                        { "header": "x-smg-target-worker", "target": true }
                    ],
                    "strip_lineage": true
                });
            }))
        );

        // The manual policy's rule says headers alone. The id and the strip
        // take the binding's block, and the rule then names no header.
        assert_eq!(
            overridden("manual").policy_document(),
            Ok(edited(preset("manual"), |document| {
                let binding = &mut document["bindings"][0];
                binding["routing_key"] = json!({
                    "sources": [{ "header": "x-smg-routing-key" }, { "rid": true }],
                    "strip_lineage": true
                });
                binding["sessions"]["sticky_header"] = json!("");
            }))
        );

        // The override places every other policy's keyed requests itself,
        // and under this one its rule has no document yet.
        assert_eq!(
            refusal(&overridden("prefix_hash")),
            ("routing_key_override.enabled", "true".to_string())
        );

        // Behind other configured headers the gateway reads the routing key
        // header as sent, and a block strips every source or none.
        let mut behind = overridden("consistent_hashing");
        behind.routing_key_override.headers = vec!["x-session-id".to_string()];
        assert_eq!(
            refusal(&behind),
            (
                "routing_key_override.headers",
                r#"["x-session-id"]"#.to_string()
            )
        );

        // The encode side reads no key and the override never places it.
        let mut plan = RoutingSettings {
            mode: ModeSettings::EncodePrefillDecode {
                encode_policy: Some(PolicySettings::RoundRobin),
                prefill_policy: Some(PolicySettings::ConsistentHashing),
                decode_policy: Some(PolicySettings::ConsistentHashing),
            },
            ..config(PolicySettings::RoundRobin)
        };
        plan.routing_key_override.enabled = true;
        let sources = plan.policy_document().map(|document| {
            document
                .bindings
                .first()
                .and_then(|binding| binding.routing_key.as_ref())
                .map(|key| key.sources.len())
        });
        assert_eq!(sources, Ok(Some(5)));
    }

    /// The policies under which the sticky override lands as a session
    /// rule: each places a single request by nothing its key hands it. The
    /// gateway's answers with the override are recorded under the first
    /// two.
    const PLACED: [&str; 7] = [
        "random",
        "power_of_two",
        "round_robin",
        "least_load",
        "passthrough",
        "bucket",
        "cache_aware",
    ];

    /// The policy of a preset's name at its flags' defaults, under the
    /// sticky override.
    fn sticky(policy: &str) -> RoutingSettings {
        let mut sticky = config(at_flag_defaults(policy));
        sticky.routing_key_override.enabled = true;
        sticky
    }

    /// Beside a policy that reads no key the override lands as a session
    /// rule on the policy's own binding: the gateway's sticky map in the
    /// document's words. The key is the configured headers, then the
    /// request's id, each without its turn and retry suffixes, and the view
    /// lists the headers, so the rule's cap reads the key's text. The
    /// controller's check takes the document, and its engine asks its host
    /// for the count of a key's requests on a worker.
    #[test]
    fn the_override_lands_as_a_session_rule_beside_a_policy_that_reads_no_key() {
        for name in PLACED {
            let expected = edited(preset(name), |document| {
                document["view"]["headers"] = json!(["x-smg-routing-key"]);
                let binding = &mut document["bindings"][0];
                binding["routing_key"] = json!({
                    "sources": [{ "header": "x-smg-routing-key" }, { "rid": true }],
                    "strip_lineage": true
                });
                binding["sessions"] = json!({
                    "ttl_s": 14_400,
                    "sweep_s": 60,
                    "assignment": "delegate",
                    "inflight_cap": 2,
                    "recovery": "latest",
                    "rule_placed": "untold"
                });
            });
            assert_eq!(sticky(name).policy_document().as_ref(), Ok(&expected));
            let applied = validated(&expected).expect("the controller takes the document");
            assert!(applied.engine().needs().session_in_flight, "{name}");
        }
    }

    /// The override's rule stands in the way of one pick. `passthrough`
    /// picks alone, so the document's answer for its place is one pick
    /// under the settings as they default. Under the override the answer
    /// names the rule: no caller is told to pick by itself past a pinned
    /// key.
    #[test]
    fn the_overrides_rule_keeps_a_place_from_one_pick() {
        let place_of = |settings: RoutingSettings| {
            let document = settings.policy_document().expect("a document");
            let applied = validated(&document).expect("the controller takes the document");
            let place = applied.engine().place_for("", Phase::Single);
            let ruled = Place::NotAlone {
                profile: "passthrough",
                in_the_way: InTheWay::Sessions,
            };
            (matches!(place, Place::Alone(_)), place == ruled)
        };
        let plain = config(at_flag_defaults("passthrough"));
        assert_eq!(place_of(plain), (true, false));
        assert_eq!(place_of(sticky("passthrough")), (false, true));
    }

    /// The override's own settings land in its rule. A key's first worker
    /// is drawn by the assignment, and the pins' order follows it: the
    /// latest worker leads under `delegate`, the first under the sticky
    /// map's own three. Either time at zero pins a key for ever. Each
    /// configured header is a source, lowercase and once, ahead of the
    /// request's id, and with none configured the id is the one source.
    #[test]
    fn the_overrides_settings_land_in_its_session_rule() {
        let written = |set: &dyn Fn(&mut RoutingKeyOverrideSettings)| {
            let mut sticky = sticky("random");
            set(&mut sticky.routing_key_override);
            let document = sticky.policy_document().expect("a document");
            assert!(validated(&document).is_ok(), "{document:?}");
            let binding = document.bindings.into_iter().next().expect("the binding");
            let rule = binding.sessions.expect("the override's rule");
            let key = binding.routing_key.expect("the key's sources");
            (rule, key, document.view.headers)
        };
        for (assignment, recovery) in [
            (Assignment::Delegate, Recovery::Latest),
            (Assignment::Random, Recovery::Original),
            (Assignment::MinLoad, Recovery::Original),
            (Assignment::MinGroup, Recovery::Original),
        ] {
            let (rule, ..) = written(&|sticky| sticky.assignment_mode = assignment);
            assert_eq!((rule.assignment, rule.recovery), (assignment, recovery));
            // The rule reads nothing of the policy beside it.
            let mut other = sticky("power_of_two");
            other.routing_key_override.assignment_mode = assignment;
            let document = other.policy_document().expect("a document");
            assert!(validated(&document).is_ok(), "{document:?}");
            let beside = document.bindings[0].sessions.as_ref();
            assert_eq!(beside, Some(&rule), "{assignment:?} beside power_of_two");
        }

        let times = |idle, interval| {
            let (rule, ..) = written(&|sticky| {
                sticky.max_idle_secs = idle;
                sticky.eviction_interval_secs = interval;
            });
            (rule.ttl_s, rule.sweep_s)
        };
        assert_eq!(times(600, 30), (600, 30));
        assert_eq!(times(0, 30), (u32::MAX, u32::MAX));
        assert_eq!(times(600, 0), (u32::MAX, u32::MAX));
        assert_eq!(times(0, 0), (u32::MAX, u32::MAX));
        let mut past = sticky("random");
        past.routing_key_override.max_idle_secs = u64::from(u32::MAX);
        assert_eq!(
            refusal(&past),
            ("routing_key_override.max_idle_secs", u32::MAX.to_string())
        );
        past.routing_key_override.max_idle_secs = 600;
        past.routing_key_override.eviction_interval_secs = u64::from(u32::MAX);
        assert_eq!(
            refusal(&past).0,
            "routing_key_override.eviction_interval_secs"
        );

        let sources = |headers: &[&str]| {
            let (_, key, view) = written(&|sticky| {
                sticky.headers = headers.iter().map(|name| (*name).to_string()).collect();
            });
            assert!(key.strip_lineage);
            let read: Vec<&str> = key
                .sources
                .iter()
                .map(|source| source.header.as_deref().unwrap_or("rid"))
                .collect();
            (read.join(" "), view.join(" "))
        };
        assert_eq!(
            sources(&["X-Session-Id", "x-smg-routing-key", "x-session-id"]),
            (
                "x-session-id x-smg-routing-key rid".to_string(),
                "x-session-id x-smg-routing-key".to_string()
            )
        );
        assert_eq!(sources(&[]), ("rid".to_string(), String::new()));
    }

    /// The override's rule places single requests alone, beside a policy
    /// that places one by nothing its key hands it. A pair plan, a gateway
    /// of many routers and the one other policy whose keyed requests the
    /// override places are each refused by the setting's name, in a
    /// sentence that says which of the three it is.
    #[test]
    fn the_override_is_refused_where_its_rule_has_no_document() {
        let why = |config: &RoutingSettings| match config.policy_document() {
            Err(RoutingSettingsError::UnsupportedSetting {
                setting: "routing_key_override.enabled",
                reason,
                ..
            }) => reason,
            other => panic!("expected the override refused by name, got {other:?}"),
        };
        for name in PLACED {
            let pair = RoutingSettings {
                mode: pair_mode(None, None),
                ..sticky(name)
            };
            assert_eq!(
                why(&pair),
                "the sticky sessions' rule places single requests alone, and this plan places a \
                 pair",
                "{name}"
            );
            assert_eq!(
                why(&many_routers(sticky(name))),
                "the sticky sessions' rule places single requests alone, and a gateway of many \
                 routers serves pairs beside them",
                "{name}"
            );
        }
        assert_eq!(
            why(&sticky("prefix_hash")),
            "under the override this policy's ring would place a key the request's own id \
             handed, and the gateway places that request by its prompt"
        );
    }

    #[test]
    fn the_sides_of_a_plan_share_one_list_of_key_sources() {
        let pair = |prefill: &str, decode: &str| RoutingSettings {
            mode: ModeSettings::PrefillDecode {
                prefill_policy: Some(at_flag_defaults(prefill)),
                decode_policy: Some(at_flag_defaults(decode)),
            },
            pd_pairing_mode: Pairing::Strict,
            ..config(PolicySettings::RoundRobin)
        };

        // One keyed side names the binding's sources, whichever side it is.
        for (prefill, decode) in [
            ("consistent_hashing", "least_load"),
            ("least_load", "consistent_hashing"),
            ("prefix_hash", "prefix_hash"),
            ("prefix_hash", "consistent_hashing"),
        ] {
            let mut expected =
                PolicySpec::pair_presets(prefill, decode).expect("a pair of presets");
            expected.plans.pd_pair.pairing = Pairing::Strict;
            assert_eq!(
                pair(prefill, decode).policy_document(),
                Ok(with_single(expected, preset("round_robin"))),
                "{prefill} and {decode}"
            );
        }

        // Two sides that read two lists are refused by their policies: with
        // no header configured the prefix ring reads none, and the hash ring
        // still reads the routing key header.
        let mut bare = pair("prefix_hash", "consistent_hashing");
        bare.routing_key_override.headers = Vec::new();
        assert_eq!(
            refusal(&bare),
            ("policy", "prefix_hash and consistent_hashing".to_string())
        );

        // The manual policy's own refusal comes first.
        assert_eq!(
            refusal(&pair("manual", "consistent_hashing")),
            ("policy", "manual".to_string())
        );
    }

    /// No document carries an overload threshold: one off its default writes
    /// nothing, alone and beside the shed word. The switch and the shed word
    /// land on the binding, each only off its default.
    #[test]
    fn an_overload_threshold_lands_on_no_binding_and_the_switch_and_the_shed_word_do() {
        let raised = RoutingSettings {
            worker_overload_waiting_requests: Some(32),
            worker_overload_token_usage: Some(0.9),
            ..config(PolicySettings::RoundRobin)
        };
        assert_eq!(raised.policy_document(), Ok(preset("round_robin")));
        let shedding = RoutingSettings {
            worker_overload_shed: true,
            ..raised
        };
        assert_eq!(
            shedding.policy_document(),
            Ok(edited(preset("round_robin"), |document| {
                document["bindings"][0]["overload"] = json!({ "shed_when_all_vetoed": true });
            })),
        );
        let off = RoutingSettings {
            worker_overload_protection: false,
            ..config(PolicySettings::RoundRobin)
        };
        assert_eq!(
            off.policy_document(),
            Ok(edited(preset("round_robin"), |document| {
                document["bindings"][0]["overload"] = json!({ "enabled": false });
            })),
        );
    }

    /// A threshold is the gateway's own beside a supplied document too: one
    /// off its default says no other document than the settings without it,
    /// so the supplied document is in force wherever it is without that
    /// threshold.
    #[test]
    fn an_overload_threshold_beside_a_given_document_is_no_disagreement() {
        let given = preset("least_load");
        for settings in [defaults(), config(at_flag_defaults("least_load"))] {
            let raised = RoutingSettings {
                policy_spec: Some(given.clone()),
                worker_overload_waiting_requests: Some(32),
                worker_overload_token_usage: Some(0.9),
                ..settings
            };
            assert_eq!(
                raised.policy_document_in_force(&defaults()),
                Ok(given.clone())
            );
        }
    }

    /// The two thresholds are no part of the comparison with a supplied
    /// document. Beside settings that state nothing else, a document is in
    /// force with one threshold at none and with both at none, whether its
    /// block is on or off: what a block that is on means over no threshold
    /// is its host's to say. The settings' own document still switches the
    /// block off for both at none.
    #[test]
    fn overload_thresholds_at_none_beside_a_given_document_are_no_disagreement() {
        let on = preset("least_load");
        let off = edited(preset("least_load"), |document| {
            document["bindings"][0]["overload"] = json!({ "enabled": false });
        });
        for (waiting, usage) in [(None, Some(0.9)), (Some(32), None), (None, None)] {
            for given in [&on, &off] {
                let beside = RoutingSettings {
                    policy_spec: Some(given.clone()),
                    worker_overload_waiting_requests: waiting,
                    worker_overload_token_usage: usage,
                    ..defaults()
                };
                assert_eq!(
                    beside.policy_document_in_force(&defaults()),
                    Ok(given.clone()),
                    "{waiting:?} and {usage:?}"
                );
            }
        }
        let own = RoutingSettings {
            worker_overload_waiting_requests: None,
            worker_overload_token_usage: None,
            ..defaults()
        };
        let block = |document: PolicySpec| {
            document
                .bindings
                .first()
                .and_then(|binding| binding.overload.clone())
        };
        let switched_off = own.policy_document_in_force(&defaults());
        assert_eq!(switched_off, own.policy_document());
        assert_eq!(switched_off.map(block), Ok(block(off)));
    }

    #[test]
    fn a_setting_no_document_says_is_refused_by_its_name() {
        let round_robin = || config(PolicySettings::RoundRobin);
        let mut sticky = config(at_flag_defaults("prefix_hash"));
        sticky.routing_key_override.enabled = true;
        let mut selection = config(at_flag_defaults("cache_aware"));
        if let PolicySettings::CacheAware {
            selection_policy, ..
        } = &mut selection.policy
        {
            *selection_policy = Some("another".to_string());
        }
        for (refused, setting, value) in [
            (sticky, "routing_key_override.enabled", "true"),
            (
                RoutingSettings {
                    dp_minimum_tokens_scheduler: true,
                    ..round_robin()
                },
                "dp_minimum_tokens_scheduler",
                "true",
            ),
            (
                RoutingSettings {
                    mode: ModeSettings::Provider("a_provider"),
                    ..round_robin()
                },
                "mode",
                "a_provider",
            ),
            (selection, "selection_policy", "another"),
        ] {
            assert_eq!(refusal(&refused), (setting, value.to_string()));
        }
    }

    /// What a host asks ahead of the document: a provider mode of a gateway
    /// of one router is named, and no other configuration is.
    #[test]
    fn a_provider_mode_of_one_router_is_named_alone() {
        let named = |mode: ModeSettings, enable_igw: bool| {
            RoutingSettings {
                mode,
                enable_igw,
                ..config(PolicySettings::RoundRobin)
            }
            .provider_alone()
        };
        let provider = || ModeSettings::Provider("a_provider");
        assert_eq!(named(provider(), false), Some("a_provider"));
        // Under the multi-router setting the document plans the gateway's
        // self-hosted workers.
        assert_eq!(named(provider(), true), None);
        for many in [false, true] {
            let pair = ModeSettings::PrefillDecode {
                prefill_policy: None,
                decode_policy: None,
            };
            let encode = ModeSettings::EncodePrefillDecode {
                encode_policy: None,
                prefill_policy: None,
                decode_policy: None,
            };
            for mode in [ModeSettings::Regular, pair, encode] {
                assert_eq!(named(mode, many), None);
            }
        }
    }

    #[test]
    fn two_sides_that_name_two_values_of_one_place_are_refused() {
        let sides = RoutingSettings {
            mode: ModeSettings::PrefillDecode {
                prefill_policy: Some(cache_aware(0.3, 64, 1.5, 120, 67_108_864)),
                decode_policy: Some(cache_aware(0.3, 64, 1.5, 60, 67_108_864)),
            },
            ..config(PolicySettings::RoundRobin)
        };
        let (setting, value) = refusal(&sides);
        assert_eq!(
            setting, "cache_index, cache_ttl_secs, max_tree_size, eviction_interval_secs",
            "the index settings stand once under a model, so two sweep intervals have no document: {value}"
        );
        // Each side's value reads as the configuration spells it.
        assert_eq!(
            value,
            r#"["tree",180,67108864,120] and ["tree",180,67108864,60]"#
        );

        // Two block sizes: the view holds one.
        let sized = |blocks: usize| {
            let mut side = cache_aware(0.3, 64, 1.5, 120, 67_108_864);
            if let PolicySettings::CacheAware { block_size, .. } = &mut side {
                *block_size = blocks;
            }
            side
        };
        let blocks = RoutingSettings {
            mode: ModeSettings::PrefillDecode {
                prefill_policy: Some(sized(16)),
                decode_policy: Some(sized(32)),
            },
            ..config(PolicySettings::RoundRobin)
        };
        assert_eq!(
            refusal(&blocks),
            ("block_size", "16 and 32".to_string()),
            "the view holds one block size, so two sides with two have no document"
        );
    }

    #[test]
    fn a_number_past_the_documents_range_is_refused() {
        let for_ever = config(PolicySettings::Manual {
            eviction_interval_secs: 60,
            max_idle_secs: u64::from(u32::MAX),
            assignment_mode: Assignment::Random,
        });
        assert_eq!(refusal(&for_ever).0, "max_idle_secs");
        let wide = config(PolicySettings::PowerOfTwo {
            load_metric: LoadMetricSetting::ResidentTokens {
                request_tokens: u64::from(u32::MAX) + 1,
            },
        });
        assert_eq!(refusal(&wide).0, "load_metric.request_tokens");
    }

    #[test]
    fn the_pin_to_an_engine_of_a_grouped_worker_moves_no_document() {
        // The dispatch pins a sticky key behind the decision, on the worker
        // the decision names: the document is the same with the pin on.
        let pinned = RoutingSettings {
            grouped_dp_sticky: Some("hash"),
            ..config(PolicySettings::RoundRobin)
        };
        assert_eq!(pinned.policy_document(), Ok(preset("round_robin")));
        assert_eq!(
            pinned.policy_document_in_force(&defaults()),
            Ok(preset("round_robin"))
        );
        // The pin reads the gateway's own key, and a supplied document names
        // its own key sources: beside one the pin is refused by its name.
        let given = RoutingSettings {
            grouped_dp_sticky: Some("hash"),
            policy_spec: Some(preset("least_load")),
            ..defaults()
        };
        assert!(matches!(
            given.policy_document_in_force(&defaults()),
            Err(RoutingSettingsError::UnsupportedSetting {
                setting: "grouped_dp_sticky",
                ..
            })
        ));
    }

    #[test]
    fn a_balance_threshold_past_the_documents_width_lands_as_its_largest_number() {
        // A threshold no load reaches: the way a gateway's own policy
        // switches a spill off. The document's largest number is one too,
        // and the controller takes it.
        let never = usize::MAX;
        let bucket = config(PolicySettings::Bucket {
            balance_abs_threshold: never,
            balance_rel_threshold: 1.5,
            bucket_adjust_interval_secs: 5,
        });
        let prefix_hash = config(PolicySettings::PrefixHash {
            prefix_token_count: 256,
            load_factor: 1.25,
            balance_abs_threshold: never,
            cache_boundaries: Vec::new(),
        });
        let cache_aware = config(cache_aware(0.3, never, 1.5, 120, 67_108_864));
        for (config, at) in [
            (bucket, "/profiles/bucket/score/0/bucket/abs_threshold"),
            (
                prefix_hash,
                "/profiles/prefix_hash/score/0/consistent-hash/bounded_load/abs_margin",
            ),
            (
                cache_aware,
                "/profiles/cache_aware/score/1/prefix-affinity/spill/abs",
            ),
        ] {
            let document = config.policy_document().expect("a document");
            let written = serde_json::to_value(&document).expect("a written form");
            assert_eq!(written.pointer(at), Some(&json!(u32::MAX)), "{at}");
            assert_eq!(validated(&document).err(), None, "{at}");
        }
    }

    #[test]
    fn an_index_the_configuration_never_sweeps_is_never_trimmed() {
        // The gateway's own policy trims its index at its sweep alone, and
        // sweeps nothing at an interval of zero: the configured size bounds
        // nothing. The document says so with its longest cadence and its
        // largest size, which no write of the controller's record meets.
        let swept_every = |secs| cache_aware(0.3, 64, 1.5, secs, 67_108_864);
        let unswept = config(swept_every(0));
        let document = unswept.policy_document().expect("a document");
        assert_eq!(
            document.index.get("*"),
            Some(&IndexSpec {
                max_size: u64::MAX,
                sweep_s: u32::MAX,
                ..IndexSpec::default()
            })
        );
        assert_eq!(validated(&document).err(), None);
        // A swept index keeps the size it is trimmed to.
        let swept = config(cache_aware(0.3, 64, 1.5, 60, 1024))
            .policy_document()
            .expect("a document");
        assert_eq!(
            swept.index.get("*"),
            Some(&IndexSpec {
                max_size: 1024,
                sweep_s: 60,
                ..IndexSpec::default()
            })
        );

        // Two sides are still compared as the configuration spells them.
        let sides = RoutingSettings {
            mode: pair_mode(Some(swept_every(0)), Some(swept_every(60))),
            ..config(PolicySettings::RoundRobin)
        };
        assert_eq!(
            refusal(&sides).1,
            r#"["tree",180,67108864,0] and ["tree",180,67108864,60]"#
        );
    }

    /// `config` as a gateway of many routers runs it.
    fn many_routers(config: RoutingSettings) -> RoutingSettings {
        RoutingSettings {
            enable_igw: true,
            ..config
        }
    }

    fn pair_mode(prefill: Option<PolicySettings>, decode: Option<PolicySettings>) -> ModeSettings {
        ModeSettings::PrefillDecode {
            prefill_policy: prefill,
            decode_policy: decode,
        }
    }

    /// The pair's own document alone, as `presets` plans it: what a pair
    /// mode says where its main policy's single binding has no document.
    fn pair_alone(config: &RoutingSettings) -> Result<PolicySpec, RoutingSettingsError> {
        presets(config).and_then(|planned| written(config, planned))
    }

    /// A document of two families: `pairs`, a pair's own document, bound over
    /// the two phases of a pair, and behind it the binding and the profile
    /// of `single`, the main policy's regular document.
    fn with_single(mut pairs: PolicySpec, single: PolicySpec) -> PolicySpec {
        for binding in &mut pairs.bindings {
            binding.matcher.phase = vec![Phase::Prefill, Phase::Decode];
        }
        pairs.profiles.extend(single.profiles);
        pairs.bindings.extend(single.bindings);
        pairs
    }

    /// A gateway in the pair mode places one single request beside its
    /// pairs: a token count takes one prefill worker, and the gateway's own
    /// routing places it by the main policy. So the mode's document binds
    /// the pair over its two phases and every other request behind it under
    /// the main policy's profile, of one router as of many, and the
    /// controller's check takes it. The pair's own document alone holds no
    /// side for a single request.
    #[test]
    fn a_pair_mode_binds_a_single_request_behind_its_pairs_under_the_main_policy() {
        let settings = round_robin_pairs_under(at_flag_defaults("least_load"));
        let document = settings
            .policy_document()
            .expect("the pair mode's document");
        assert_eq!(
            serde_json::to_value(&document.bindings).expect("a written form"),
            json!([
                {
                    "match": { "model": "*", "phase": ["prefill", "decode"] },
                    "plan": "pd-pair",
                    "profiles": { "decode": "decode", "prefill": "prefill" },
                },
                { "match": { "model": "*", "phase": [] }, "profile": "least_load" },
            ]),
        );
        assert_eq!(
            many_routers(settings.clone()).policy_document().as_ref(),
            Ok(&document)
        );

        // Whether a single request has no side, and whether its place is
        // the main policy's profile.
        let single_request = |document: &PolicySpec| {
            let applied = validated(document).expect("the controller takes the document");
            let place = applied.engine().place_for("", Phase::Single);
            let main = Place::NotAlone {
                profile: "least_load",
                in_the_way: InTheWay::Scorer("least-load"),
            };
            (place == Place::NoSide, place == main)
        };
        let alone = pair_alone(&settings).expect("the pair's own document");
        assert_eq!(single_request(&alone), (true, false));
        assert_eq!(single_request(&document), (false, true));
    }

    /// A pair mode whose main policy states a setting no document says
    /// plans its pairs alone: the single binding is left out, of one router
    /// as of many, and a single request has no side there.
    #[test]
    fn a_pair_mode_plans_pairs_alone_where_its_main_policy_has_no_document() {
        // Under the override the hash ring on both sides of the pair has a
        // document, and a main policy of `prefix_hash` has none for its
        // keyed requests.
        let mut keyed = RoutingSettings {
            mode: pair_mode(
                Some(PolicySettings::ConsistentHashing),
                Some(PolicySettings::ConsistentHashing),
            ),
            ..config(at_flag_defaults("prefix_hash"))
        };
        keyed.routing_key_override.enabled = true;
        let alone = pair_alone(&keyed).expect("the pair's own document");
        assert_eq!(keyed.policy_document().as_ref(), Ok(&alone));
        assert_eq!(alone.bindings.len(), 1);
        let applied = validated(&alone).expect("the controller takes the document");
        assert_eq!(applied.engine().place_for("", Phase::Single), Place::NoSide);

        // Under a main policy the override has a rule for, the same pair
        // binds its single request, and the rule is the single binding's.
        keyed.policy = PolicySettings::Random;
        let both = keyed
            .policy_document()
            .expect("the document of two families");
        let rules: Vec<bool> = both
            .bindings
            .iter()
            .map(|binding| binding.sessions.is_some())
            .collect();
        assert_eq!(rules, [false, true]);
        assert_eq!(validated(&both).err(), None);
    }

    /// The view holds one block size, and a prefix ring counts its key in
    /// it without naming one. A main policy that names another size beside a
    /// pair of prefix rings would end their key at another token (blocks of
    /// 96 under a depth of 256) or leave it no whole block (blocks of 512),
    /// and a pair that names the size does the same to a main policy's
    /// ring. Either way a gateway of one router plans the pair alone, as the
    /// pair's own document says it, and the controller's check takes it. A
    /// gateway of many routers serves each family as a fleet of its own and
    /// keeps the two bindings it had. At the ring's own size the single
    /// request is bound.
    #[test]
    fn a_pair_mode_plans_pairs_alone_where_a_block_size_would_move_a_rings_key() {
        let ring = || at_flag_defaults("prefix_hash");
        let both_ways = |blocks: usize| {
            [
                RoutingSettings {
                    mode: pair_mode(Some(ring()), Some(ring())),
                    ..config(cache_aware_over(blocks))
                },
                RoutingSettings {
                    mode: pair_mode(
                        Some(cache_aware_over(blocks)),
                        Some(cache_aware_over(blocks)),
                    ),
                    ..config(ring())
                },
            ]
        };
        for blocks in [96, 512] {
            for settings in both_ways(blocks) {
                let alone = pair_alone(&settings).expect("the pair's own document");
                assert_eq!(validated(&alone).err(), None, "{blocks}");
                assert_eq!(settings.policy_document().as_ref(), Ok(&alone), "{blocks}");
                assert_eq!(
                    many_routers(settings)
                        .policy_document()
                        .map(|both| both.bindings.len()),
                    Ok(2),
                    "{blocks}"
                );
            }
        }
        for settings in both_ways(16) {
            let both = settings
                .policy_document()
                .expect("the document of two families");
            assert_eq!(both.bindings.len(), 2);
            assert_eq!(validated(&both).err(), None);
        }
        // The sides of one binding say one document together: a ring beside
        // a side that names the size counts in it, as it did, and the
        // single request is bound under a main policy that names none.
        let mixed = RoutingSettings {
            mode: pair_mode(Some(ring()), Some(cache_aware_over(96))),
            ..config(PolicySettings::RoundRobin)
        };
        let both = mixed
            .policy_document()
            .expect("the document of two families");
        assert_eq!((both.bindings.len(), both.view.block_size), (2, Some(96)));
        assert_eq!(validated(&both).err(), None);
    }

    /// Over gRPC or ZMQ a gateway in the pair mode counts tokens with no
    /// worker. Its document of one router is then the pair's own: it binds
    /// no single request, and reads nothing for the main policy. A gateway
    /// of many routers serves single workers however it reaches them, and
    /// over HTTP the single request is bound. A given document beside no
    /// routing setting is in force over each.
    #[test]
    fn a_pair_mode_of_one_router_binds_a_single_request_over_http_alone() {
        let over = |connection_mode| RoutingSettings {
            connection_mode,
            ..round_robin_pairs_under(at_flag_defaults("cache_aware"))
        };
        let events = |document: &PolicySpec| {
            validated(document)
                .map(|applied| applied.needs().events)
                .ok()
        };
        let both = over(ConnectionModeSetting::Http)
            .policy_document()
            .expect("the document over HTTP");
        assert_eq!((both.bindings.len(), events(&both)), (2, Some(true)));

        let given = preset("round_robin");
        for connection_mode in [ConnectionModeSetting::Grpc, ConnectionModeSetting::Zmq] {
            let settings = over(connection_mode);
            let alone = pair_alone(&settings).expect("the pair's own document");
            assert_eq!(
                (alone.bindings.len(), events(&alone)),
                (1, Some(false)),
                "{connection_mode:?}"
            );
            assert_eq!(
                settings.policy_document().as_ref(),
                Ok(&alone),
                "{connection_mode:?}"
            );
            assert_eq!(
                many_routers(settings).policy_document().as_ref(),
                Ok(&both),
                "{connection_mode:?}"
            );
            let unset = RoutingSettings {
                mode: pair_mode(None, None),
                connection_mode,
                policy_spec: Some(given.clone()),
                ..defaults()
            };
            assert_eq!(
                unset.policy_document_in_force(&defaults()),
                Ok(given.clone()),
                "{connection_mode:?}"
            );
        }
    }

    #[test]
    fn the_default_document_of_many_routers_binds_pairs_ahead_of_single_workers() {
        let document = many_routers(defaults())
            .policy_document()
            .expect("the default document");
        assert_eq!(
            serde_json::to_value(&document.bindings).expect("a written form"),
            json!([
                {
                    "match": { "model": "*", "phase": ["prefill", "decode"] },
                    "plan": "pd-pair",
                    "profiles": { "decode": "decode", "prefill": "prefill" },
                },
                { "match": { "model": "*", "phase": [] }, "profile": "random" },
            ]),
        );
        assert_eq!(
            document.profiles.keys().collect::<Vec<_>>(),
            ["decode", "prefill", "random"]
        );
    }

    #[test]
    fn many_routers_say_each_family_as_its_own_configuration_does() {
        for name in PRESETS.iter().copied().filter(|name| *name != "manual") {
            let regular = config(at_flag_defaults(name));
            let pair = RoutingSettings {
                mode: pair_mode(None, None),
                ..regular.clone()
            };
            let single = regular.policy_document().expect("the regular document");
            let paired = pair_alone(&pair).expect("the pair's own document");
            let both = many_routers(regular.clone())
                .policy_document()
                .expect("the document of many routers");

            // The pair binding leads over its two phases. The single binding
            // stands behind it as the regular configuration writes it.
            let mut bindings = paired.bindings.clone();
            for binding in &mut bindings {
                binding.matcher.phase = vec![Phase::Prefill, Phase::Decode];
            }
            bindings.extend(single.bindings.clone());
            assert_eq!(both.bindings, bindings, "{name}");
            let mut profiles = paired.profiles.clone();
            profiles.extend(single.profiles.clone());
            assert_eq!(both.profiles, profiles, "{name}");
            assert_eq!(
                (&both.view, &both.index),
                (&single.view, &single.index),
                "{name}"
            );

            // A pair configuration with no side policy of its own says the
            // same document, of one router as of many, and the controller
            // takes it.
            assert_eq!(pair.policy_document().as_ref(), Ok(&both), "{name}");
            assert_eq!(
                many_routers(pair).policy_document().as_ref(),
                Ok(&both),
                "{name}"
            );
            assert_eq!(validated(&both).err(), None, "{name}");
        }
    }

    #[test]
    fn a_setting_of_the_binding_lands_on_both_bindings_of_many_routers() {
        let mut keyed = RoutingSettings {
            worker_overload_waiting_requests: Some(32),
            worker_overload_token_usage: Some(0.9),
            worker_overload_shed: true,
            pd_pairing_mode: Pairing::Strict,
            ..config(PolicySettings::ConsistentHashing)
        };
        keyed.routing_key_override.headers = vec!["x-user".to_string()];
        let single = keyed.policy_document().expect("the regular document");
        let (overload, key) = single
            .bindings
            .first()
            .map(|binding| (binding.overload.clone(), binding.routing_key.clone()))
            .expect("the regular binding");
        // The block states its shed word and no number: a threshold off its
        // default lands on no binding.
        let sheds = OverloadSpec {
            shed_when_all_vetoed: true,
            ..OverloadSpec::default()
        };
        assert_eq!(overload, Some(sheds));
        assert!(key.is_some());

        let both = many_routers(keyed)
            .policy_document()
            .expect("the document of many routers");
        assert_eq!(both.bindings.len(), 2);
        for binding in &both.bindings {
            // One switch and one shed word stand for the pools of each
            // family, and each family reads a request's key from the same
            // sources.
            assert_eq!(binding.overload, overload);
            assert_eq!(binding.routing_key, key);
        }
        // The pairing mode is the pair plan's under a regular configuration too.
        assert_eq!(both.plans.pd_pair.pairing, Pairing::Strict);
    }

    #[test]
    fn a_given_document_is_compared_with_the_document_of_many_routers() {
        let given = preset("round_robin");
        for mode in [ModeSettings::Regular, pair_mode(None, None)] {
            // No routing setting is set: the given document is in force.
            let unset = many_routers(RoutingSettings {
                mode: mode.clone(),
                policy_spec: Some(given.clone()),
                ..defaults()
            });
            assert_eq!(
                unset.policy_document_in_force(&defaults()),
                Ok(given.clone())
            );

            // A setting that says another document is refused, and one that
            // says the given document is not.
            let set = many_routers(RoutingSettings {
                mode,
                ..config(PolicySettings::RoundRobin)
            });
            let said = set.policy_document().expect("the flags' document");
            let other = RoutingSettings {
                policy_spec: Some(given.clone()),
                ..set.clone()
            };
            assert!(matches!(
                other.policy_document_in_force(&defaults()),
                Err(RoutingSettingsError::Disagreement { .. })
            ));
            let same = RoutingSettings {
                policy_spec: Some(said.clone()),
                ..set
            };
            assert_eq!(same.policy_document_in_force(&defaults()), Ok(said));
        }
    }

    /// A pair mode of one router says two bindings, so a given document is
    /// compared with both. Beside a stated policy, the pair's own document
    /// is refused at the single request's profile, which it lacks, and the
    /// document of two bindings is in force. Where no routing setting is
    /// set, the pair's own document is in force.
    #[test]
    fn a_given_document_is_compared_with_the_two_bindings_of_a_pair_mode() {
        let stated = round_robin_pairs_under(at_flag_defaults("least_load"));
        let said = stated.policy_document().expect("the mode's document");
        let alone = pair_alone(&stated).expect("the pair's own document");
        assert_eq!((said.bindings.len(), alone.bindings.len()), (2, 1));
        let given = |document: &PolicySpec, settings: &RoutingSettings| RoutingSettings {
            policy_spec: Some(document.clone()),
            ..settings.clone()
        };

        let refused = given(&alone, &stated).policy_document_in_force(&defaults());
        assert!(
            matches!(
                &refused,
                Err(RoutingSettingsError::Disagreement { at, document, .. })
                    if at == "profiles.least_load" && document == "nothing"
            ),
            "{refused:?}"
        );
        assert_eq!(
            given(&said, &stated).policy_document_in_force(&defaults()),
            Ok(said)
        );
        let unset = RoutingSettings {
            mode: pair_mode(None, None),
            ..defaults()
        };
        assert_eq!(
            given(&alone, &unset).policy_document_in_force(&defaults()),
            Ok(alone)
        );
    }

    #[test]
    fn many_routers_plan_one_family_where_the_other_has_no_document() {
        // The manual policy runs on no side of a pair.
        let manual = config(at_flag_defaults("manual"));
        assert!(manual.policy_document().is_ok());
        assert_eq!(
            many_routers(manual.clone()).policy_document(),
            manual.policy_document()
        );

        // Under the override the main policy's keyed requests have no
        // document yet, while the hash ring on both sides of the pair has
        // one. The pair configuration's document then plans pairs alone.
        let mut keyed = RoutingSettings {
            mode: pair_mode(
                Some(PolicySettings::ConsistentHashing),
                Some(PolicySettings::ConsistentHashing),
            ),
            ..config(PolicySettings::RoundRobin)
        };
        keyed.routing_key_override.enabled = true;
        keyed.routing_key_override.headers = vec![MANUAL_SESSION_HEADER.to_string()];
        let pairs_alone = pair_alone(&keyed);
        assert!(pairs_alone.is_ok(), "{pairs_alone:?}");
        assert_eq!(many_routers(keyed.clone()).policy_document(), pairs_alone);
        // The same main policy alone is refused by the setting's name.
        keyed.mode = ModeSettings::Regular;
        assert_eq!(
            refusal(&many_routers(keyed)),
            ("routing_key_override.enabled", "true".to_string())
        );
    }

    /// A gateway of many routers in the encode mode serves single workers
    /// beside its encoders and pairs: a model of single workers is placed
    /// by the main policy. So its document binds the mode's three sides
    /// over their phases and every other request behind them, in that
    /// order: the first binding that matches a request decides it.
    #[test]
    fn the_encode_mode_of_many_routers_binds_a_single_request_behind_its_sides() {
        let encode_mode = |encode: Option<PolicySettings>, sides: Option<PolicySettings>| {
            ModeSettings::EncodePrefillDecode {
                encode_policy: encode,
                decode_policy: sides.clone(),
                prefill_policy: sides,
            }
        };
        let settings = many_routers(RoutingSettings {
            mode: encode_mode(None, None),
            ..config(PolicySettings::RoundRobin)
        });
        let document = settings.policy_document().expect("the mode's document");
        assert_eq!(
            serde_json::to_value(&document.bindings).expect("a written form"),
            json!([
                {
                    "match": { "model": "*", "phase": ["encode", "prefill", "decode"] },
                    "plan": "epd",
                    "profiles": { "decode": "decode", "encode": "encode", "prefill": "prefill" },
                },
                { "match": { "model": "*", "phase": [] }, "profile": "round_robin" },
            ]),
        );
        let applied = validated(&document).expect("the controller takes the document");
        assert_ne!(applied.engine().place_for("", Phase::Single), Place::NoSide);

        // A prefix ring counts its key in its own binding's blocks. Where
        // the main policy names another size, the mode's own document
        // stands, as it does under one router, and the controller takes it.
        let ringed = RoutingSettings {
            mode: encode_mode(None, Some(at_flag_defaults("prefix_hash"))),
            ..config(cache_aware_over(512))
        };
        let own = ringed.policy_document().expect("the mode's own document");
        assert!(validated(&own).is_ok());
        assert_eq!(many_routers(ringed).policy_document().as_ref(), Ok(&own));

        // No configuration of the mode loses its document to its single
        // request. Every policy is the main one in turn, the encode side
        // and the pair's sides are unnamed or of one policy, and the sticky
        // override is off and on. Where the mode's own document, as
        // `presets` plans it, is refused, the settings' is refused by the
        // same sentence. Where the controller's check takes the mode's
        // own, it takes the settings', which binds a single request behind
        // the mode's sides in all but these: under the override, beside
        // sides of the hash ring, every main policy but
        // `consistent_hashing` and `manual` has no document for its keyed
        // requests, and the mode is planned alone, whatever its encode
        // side.
        let unnamed_or_each = || std::iter::once(None).chain(PRESETS.iter().copied().map(Some));
        let mut alone_still = Vec::new();
        for main in PRESETS {
            for encode in unnamed_or_each() {
                for sides in unnamed_or_each() {
                    for sticky in [false, true] {
                        let named = format!("{main} over {encode:?} and {sides:?}, {sticky}");
                        let mut swept = many_routers(RoutingSettings {
                            mode: encode_mode(
                                encode.map(at_flag_defaults),
                                sides.map(at_flag_defaults),
                            ),
                            ..config(at_flag_defaults(main))
                        });
                        swept.routing_key_override.enabled = sticky;
                        let alone = presets(&swept).and_then(|planned| written(&swept, planned));
                        let said = swept.policy_document();
                        let (Ok(one), Ok(both)) = (&alone, &said) else {
                            assert_eq!(said, alone, "{named}");
                            continue;
                        };
                        let taken = validated(one).is_ok();
                        assert_eq!(validated(both).is_ok(), taken, "{named}");
                        if taken && both.bindings.len() == 1 {
                            assert_eq!(both, one, "{named}");
                            alone_still.push((main, sides, sticky));
                        }
                    }
                }
            }
        }
        alone_still.dedup();
        let ring = Some("consistent_hashing");
        let keyless = [
            "random",
            "round_robin",
            "passthrough",
            "power_of_two",
            "least_load",
            "prefix_hash",
            "cache_aware",
            "bucket",
        ];
        assert_eq!(alone_still, keyless.map(|main| (main, ring, true)));
    }

    /// A pair configuration with round robin on both sides, under `main`.
    fn round_robin_pairs_under(main: PolicySettings) -> RoutingSettings {
        RoutingSettings {
            mode: pair_mode(
                Some(PolicySettings::RoundRobin),
                Some(PolicySettings::RoundRobin),
            ),
            ..config(main)
        }
    }

    /// The cache-aware policy at its flags' defaults, over blocks of `blocks`.
    fn cache_aware_over(blocks: usize) -> PolicySettings {
        let mut policy = at_flag_defaults("cache_aware");
        if let PolicySettings::CacheAware { block_size, .. } = &mut policy {
            *block_size = blocks;
        }
        policy
    }

    #[test]
    fn many_routers_write_the_view_and_index_of_a_main_policy_no_side_runs() {
        // Round robin places both sides of a pair and names no view and no
        // index. A tuned cache-aware policy places single workers: what it
        // shares with every side lands in the document all the same.
        let mut main = cache_aware(0.3, 64, 1.5, 60, 16_777_216);
        if let PolicySettings::CacheAware { block_size, .. } = &mut main {
            *block_size = 256;
        }
        let single = config(main.clone())
            .policy_document()
            .expect("the regular document");
        assert!(single.view.block_size.is_some() && !single.index.is_empty());
        let pairs = round_robin_pairs_under(main);
        let paired = pair_alone(&pairs).expect("the pair's own document");
        let both = many_routers(pairs.clone())
            .policy_document()
            .expect("the document of many routers");

        assert_eq!(both.bindings.len(), 2);
        assert_eq!(both.bindings.last(), single.bindings.first());
        assert_eq!(
            both.profiles.get("cache_aware"),
            single.profiles.get("cache_aware")
        );
        for side in ["prefill", "decode"] {
            assert_eq!(both.profiles.get(side), paired.profiles.get(side), "{side}");
        }
        assert_eq!((&both.view, &both.index), (&single.view, &single.index));
        assert_eq!(validated(&both).err(), None);
    }

    #[test]
    fn many_routers_keep_the_manual_rule_on_the_single_binding_alone() {
        // The manual policy places single workers while round robin places
        // both sides of a pair. The session rule and the view's header are
        // the single binding's, and the pair binding holds no rule.
        let manual = at_flag_defaults("manual");
        let single = config(manual.clone())
            .policy_document()
            .expect("the regular document");
        let pairs = round_robin_pairs_under(manual);
        let both = many_routers(pairs.clone())
            .policy_document()
            .expect("the document of many routers");

        assert_eq!(both.bindings.len(), 2);
        let pair = both.bindings.first().expect("the pair binding");
        assert_eq!((&pair.sessions, &pair.routing_key), (&None, &None));
        let rule = both.bindings.last().expect("the single binding");
        assert!(rule.sessions.is_some());
        assert_eq!(Some(rule), single.bindings.first());
        assert_eq!(both.view, single.view);
        assert!(!both.view.headers.is_empty());
        assert_eq!(validated(&both).err(), None);
    }

    #[test]
    fn many_routers_plan_pairs_alone_where_the_main_policy_names_another_shared_value() {
        // The view holds one block size. The sides agree on theirs and the
        // main policy names another, so single workers have no document and
        // the pair configuration plans its own family alone.
        let clash = RoutingSettings {
            mode: pair_mode(Some(cache_aware_over(16)), Some(cache_aware_over(16))),
            ..config(cache_aware_over(32))
        };
        let pairs_alone = pair_alone(&clash);
        assert!(pairs_alone.is_ok(), "{pairs_alone:?}");
        assert_eq!(clash.policy_document(), pairs_alone);
        assert_eq!(many_routers(clash).policy_document(), pairs_alone);
    }

    /// No pair mode loses its document to its single request. Every policy
    /// is the main one in turn, the sides are unnamed or both of one policy,
    /// and the sticky override is off and on. Where the pair's own document
    /// is refused, the mode's is refused by the same sentence. Where the
    /// controller's check takes the pair's own document, it takes the
    /// mode's, which binds a single request behind the pair in all but
    /// two: under the override, beside sides of the hash ring or of the
    /// manual policy, a main policy of `prefix_hash` has no document for
    /// its keyed requests, and the pair is planned alone.
    #[test]
    fn no_pair_mode_loses_its_document_to_its_single_request() {
        let mut alone_still = Vec::new();
        for main in PRESETS {
            for sides in std::iter::once(None).chain(PRESETS.iter().copied().map(Some)) {
                for sticky in [false, true] {
                    let mut settings = RoutingSettings {
                        mode: pair_mode(sides.map(at_flag_defaults), sides.map(at_flag_defaults)),
                        ..config(at_flag_defaults(main))
                    };
                    settings.routing_key_override.enabled = sticky;
                    let named = format!("{main} over {sides:?}, the override {sticky}");
                    let (alone, said) = (pair_alone(&settings), settings.policy_document());
                    let (Ok(own), Ok(both)) = (&alone, &said) else {
                        assert_eq!(said, alone, "{named}");
                        continue;
                    };
                    let taken = validated(own).is_ok();
                    assert_eq!(validated(both).is_ok(), taken, "{named}");
                    if taken && both.bindings.len() == 1 {
                        assert_eq!(both, own, "{named}");
                        alone_still.push((main, sides, sticky));
                    }
                }
            }
        }
        let (ring, manual) = (Some("consistent_hashing"), Some("manual"));
        assert_eq!(
            alone_still,
            [("prefix_hash", ring, true), ("prefix_hash", manual, true)]
        );
    }
}
