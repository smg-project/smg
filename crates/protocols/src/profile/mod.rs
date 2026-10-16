//! Per-provider protocol profiles.
//!
//! A profile owns the request rules a provider's vendor-acceptance contract
//! enforces beyond (or instead of) the OpenAI baseline. Profiles are selected
//! from the request's model id and applied during request validation, so every
//! entry point using `ValidatedJson` gets them for free.
//!
//! Precedence for what a profile encodes: provider verifier > vendor manual >
//! live API behavior.
//!
//! A profile also shapes the request before validation: message-level
//! extension structs that belong to another provider are dropped (see
//! [`crate::ext::ProviderExt`]). Provider fields typed directly onto content
//! parts, such as MiniMax's `max_long_side_pixel` and `fps`, are not covered
//! by that pass and are forwarded as sent.

mod deepseek;
mod kimi;
mod minimax;
mod openai;
mod zai;

use std::{
    collections::HashMap,
    sync::{Arc, OnceLock, RwLock},
};

use crate::{
    chat::{thinking_from_reasoning_effort, ChatCompletionRequest, ChatMessage},
    common::{invalid, Tool},
    ext::retain_if,
};

/// What the configured model aliases record for a name. An alias is resolved
/// into the request's model id before the response side reads it, so the
/// served name has to select the profile its vendor-named alias does, or a
/// request that entered under the vendor name loses the profile half-way; and
/// a profile's reading of a version from the model name (DeepSeek V4 against
/// V4.1) goes through the same record ([`served_under`]), so a request under
/// an opaque alias of a versioned model is read as that model.
enum Registered {
    /// A configured alias: the served model id it resolves to, that model's
    /// profile, recorded so a request under the alias finds it in one step,
    /// and whether the alias's own name names the vendor whose contract it
    /// resolves to (such an alias is read as its own name, the way a request
    /// naming it always was; an opaque alias, and one vendor's name for
    /// another vendor's model, are read as the served model).
    Alias {
        served: Arc<str>,
        profile: ProviderProfile,
        own_contract: bool,
    },
    /// A served model whose own name selects no vendor contract and which
    /// takes one from the vendor-named aliases it is served under: the
    /// profile, and every alias of that vendor. A version read from the name
    /// consults them all and holds only when they agree, so the reading does
    /// not depend on the order the aliases were configured in, and an
    /// ambiguous configuration (one checkpoint under the names of two
    /// versions) leaves the opaque name the reading it had before any alias.
    Contract {
        profile: ProviderProfile,
        vendor_names: Vec<Arc<str>>,
    },
}

/// The registered names, recorded once at startup from the configured model
/// aliases ([`ProviderProfile::register_model_aliases`]).
fn registry() -> &'static RwLock<HashMap<String, Registered>> {
    static REGISTRY: OnceLock<RwLock<HashMap<String, Registered>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Whether `model` is served under a name that satisfies `names`. A name
/// the table does not know, and an alias whose own name names the vendor it
/// resolves to, are read as they are: a request naming them is read the way
/// it always was. Any other alias (an opaque name, or one vendor's name for
/// another vendor's model) is read as the served model it resolves to, and
/// an opaque served model as the vendor-named aliases it is served under,
/// all of which have to agree. A profile reads a version from the model name through
/// this, so a request under an opaque alias of a versioned model is read as
/// that model, and a served model under one version's names as that version
/// whichever of its names the request carries.
fn served_under(model: &str, names: impl Fn(&str) -> bool) -> bool {
    let registry = registry().read().unwrap_or_else(|e| e.into_inner());
    let agree = |vendor_names: &[Arc<str>]| vendor_names.iter().all(|name| names(name));
    match registry.get(model) {
        None
        | Some(Registered::Alias {
            own_contract: true, ..
        }) => names(model),
        Some(Registered::Alias { served, .. }) => match registry.get(served.as_ref()) {
            Some(Registered::Contract { vendor_names, .. }) => agree(vendor_names),
            _ => names(served),
        },
        Some(Registered::Contract { vendor_names, .. }) => agree(vendor_names),
    }
}

/// Provider dialect for a request, selected from the model id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderProfile {
    /// The default: a model whose id selects no vendor contract, which is
    /// every self-hosted deployment. Only the structural rules of core
    /// validation apply; strict schemas, `json_object` prompts and tool-message
    /// pairing go through to the engine, whose grammar compiler validates
    /// what it needs.
    Generic,
    /// OpenAI's own hosted models ([`is_openai_vendor_model`]): the public
    /// API's structured-output and tool-message rules on top of core
    /// validation.
    OpenAi,
    /// Kimi/Moonshot contract (Kimi-Vendor-Verifier).
    Kimi,
    /// MiniMax contract (MiniMax-Provider-Verifier).
    Minimax,
    /// z.ai / GLM contract (providers-verifier `golden/zai`).
    Zai,
    /// DeepSeek V4 / V4.1 Chat contract (deepseek-provider-verifier).
    DeepSeek,
}

impl ProviderProfile {
    /// Tools the profile lets messages declare on top of the request-level
    /// `tools`: Kimi K3's dynamic tools on system and developer messages, in
    /// message order. Empty for profiles without message-declared tools, so
    /// a foreign `tools` field on a message never widens the tool set.
    pub fn dynamic_tools<'a>(
        self,
        req: &'a ChatCompletionRequest,
    ) -> Box<dyn Iterator<Item = &'a Tool> + 'a> {
        match self {
            ProviderProfile::Kimi => Box::new(kimi::dynamic_tools(req)),
            ProviderProfile::DeepSeek
            | ProviderProfile::Minimax
            | ProviderProfile::Zai
            | ProviderProfile::OpenAi
            | ProviderProfile::Generic => Box::new(std::iter::empty()),
        }
    }

    /// Whether responses are scanned for tool calls even when the request
    /// declares no tools. MiniMax's verifier expects a tool the conversation
    /// established (a retry after a transient tool error, say) to come back
    /// as a `tool_calls` finish without any tool inventory; the model's
    /// tool-call markup is a dedicated token, so scanning every response is
    /// unambiguous. Other profiles keep the parser gated on declared tools.
    pub fn parses_tool_calls_without_tools(self) -> bool {
        matches!(self, ProviderProfile::Minimax)
    }

    /// Select the profile for a model id.
    ///
    /// A registered alias or served model id takes the profile recorded for
    /// it at startup ([`Self::register_model_aliases`]): an alias resolves to
    /// its served model's profile, so the profile is the same before and
    /// after the alias is resolved into the served name. Otherwise the name
    /// decides, the way the tool and reasoning parser factories match: any
    /// `/`-separated segment that starts with a vendor marker selects the
    /// profile, so `kimi-k3`, `/models/Kimi-K3`, `moonshotai/kimi-k2` and
    /// `openrouter/moonshotai/kimi-k2` all resolve to Kimi; OpenAI's own
    /// model ids ([`is_openai_vendor_model`]: `gpt-4o`, `openai/gpt-4.1`,
    /// `chatgpt-4o-latest`, `o3-mini`) resolve to OpenAI. Any other id is the
    /// default profile: a self-hosted model, whose extensions from another
    /// vendor are dropped with a warning and whose `root` message is rejected
    /// outright (that role needs a MiniMax model id or alias). DeepSeek is
    /// narrower: only the calibrated V4 / V4.1 model segments and the
    /// `deepseek-flash` alias select its profile; older versions and
    /// unrecognized suffixes keep the default.
    pub fn for_model(model: &str) -> Self {
        let registered = registry()
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(model)
            .map(|registered| match registered {
                Registered::Alias { profile, .. } | Registered::Contract { profile, .. } => {
                    *profile
                }
            });
        registered.unwrap_or_else(|| Self::from_model_segments(model))
    }

    /// The thinking on/off preference the profile reads from a request's
    /// effective `reasoning_effort` (`thinking.effort`, else the top-level
    /// field) when the request states no `thinking.type`. The OpenAI reading
    /// ([`thinking_from_reasoning_effort`]: `none` and `minimal` switch
    /// thinking off, every other value leaves the template's default) for
    /// every profile but MiniMax, whose contract keeps thinking on for every
    /// value: `none` is the model's default depth and `minimal` is `low`.
    pub fn thinking_from_reasoning_effort(self, reasoning_effort: Option<&str>) -> Option<bool> {
        match self {
            ProviderProfile::Minimax => None,
            _ => thinking_from_reasoning_effort(reasoning_effort),
        }
    }

    /// Whether an explicit `reasoning_effort` outranks `thinking.type:
    /// disabled`: under the MiniMax contract it does (an effort asks for
    /// thinking at that depth); every other profile honours the toggle.
    pub fn effort_outranks_thinking_disabled(self) -> bool {
        matches!(self, ProviderProfile::Minimax)
    }

    /// Record the configured model aliases (`alias -> served model id`).
    ///
    /// A served model id whose own name selects no vendor contract takes the
    /// contract of a vendor-named alias (Kimi, MiniMax, z.ai, DeepSeek) it is
    /// served under, whichever of its names a request carries; when two
    /// aliases of one served model name different vendors, the first wins and
    /// the conflict is logged. An OpenAI-looking alias (`gpt-4o=local-model`)
    /// is compatibility naming for OpenAI clients, not a contract: it binds
    /// nothing, and because every alias resolves to its served model's
    /// profile, a request naming such an alias is judged as the self-hosted
    /// model it reaches, not under the public API's rules.
    ///
    /// The record also keeps which names carry a model's contract (the
    /// vendor-named aliases of a served model whose own name selects none),
    /// so a profile that reads a version from the model name reads an
    /// opaque name as the vendor names behind it (`served_under`); an
    /// alias named for the vendor it resolves to is read as its own name,
    /// as it always was, and one vendor's name for another vendor's model
    /// as that model.
    pub fn register_model_aliases<'a>(aliases: impl IntoIterator<Item = (&'a str, &'a str)>) {
        let aliases: Vec<(&str, &str)> = aliases.into_iter().collect();
        let mut map = registry().write().unwrap_or_else(|e| e.into_inner());
        // First the served models' contracts, from their own names or their
        // vendor-named aliases ...
        for (alias, canonical) in &aliases {
            let own = Self::from_model_segments(canonical);
            if own.is_vendor_contract() {
                continue;
            }
            let profile = Self::from_model_segments(alias);
            if !profile.is_vendor_contract() {
                continue;
            }
            match map.get_mut(*canonical) {
                Some(Registered::Contract {
                    profile: existing,
                    vendor_names,
                }) => {
                    if *existing != profile {
                        tracing::warn!(
                            canonical,
                            alias,
                            kept = ?existing,
                            ignored = ?profile,
                            "model alias names a different vendor than an earlier alias of the same served model"
                        );
                    } else if !vendor_names.iter().any(|name| name.as_ref() == *alias) {
                        vendor_names.push(Arc::from(*alias));
                    }
                }
                // An alias of an alias names no served model: it binds
                // nothing.
                Some(Registered::Alias { .. }) => {}
                None => {
                    map.insert(
                        (*canonical).to_string(),
                        Registered::Contract {
                            profile,
                            vendor_names: vec![Arc::from(*alias)],
                        },
                    );
                }
            }
        }
        // ... then every alias resolves to its served model's profile.
        for (alias, canonical) in &aliases {
            let profile = match map.get(*canonical) {
                Some(Registered::Alias { profile, .. } | Registered::Contract { profile, .. }) => {
                    *profile
                }
                None => Self::from_model_segments(canonical),
            };
            // The alias's own name carries the contract only when it names
            // the vendor the alias resolves to.
            let own = Self::from_model_segments(alias);
            map.insert(
                (*alias).to_string(),
                Registered::Alias {
                    served: Arc::from(*canonical),
                    profile,
                    own_contract: own.is_vendor_contract() && own == profile,
                },
            );
        }
    }

    /// Whether this is a vendor's contract (Kimi, MiniMax, z.ai, DeepSeek)
    /// that an alias may bind to a served model; the default profile and
    /// OpenAI's are not: the former is the absence of a contract, the latter
    /// follows OpenAI's own model names only.
    fn is_vendor_contract(self) -> bool {
        matches!(
            self,
            ProviderProfile::Kimi
                | ProviderProfile::Minimax
                | ProviderProfile::Zai
                | ProviderProfile::DeepSeek
        )
    }

    /// The vendor a model id names in one of its `/`-separated segments.
    fn from_model_segments(model: &str) -> Self {
        for segment in model.split('/') {
            if deepseek::matches_model(segment) {
                return ProviderProfile::DeepSeek;
            }
            if starts_with_ignore_ascii_case(segment, "kimi")
                || starts_with_ignore_ascii_case(segment, "moonshot")
            {
                return ProviderProfile::Kimi;
            }
            if starts_with_ignore_ascii_case(segment, "minimax")
                || starts_with_ignore_ascii_case(segment, "abab")
            {
                return ProviderProfile::Minimax;
            }
            if starts_with_ignore_ascii_case(segment, "glm")
                || starts_with_ignore_ascii_case(segment, "zai")
                || starts_with_ignore_ascii_case(segment, "z-ai")
            {
                return ProviderProfile::Zai;
            }
        }
        if is_openai_vendor_model(model) {
            return ProviderProfile::OpenAi;
        }
        ProviderProfile::Generic
    }

    /// Whether [`Self::normalize_chat`] shapes a chat request under this
    /// profile beyond the foreign-extension drop every profile runs: the
    /// vendor contracts fill their defaults (a stream's usage chunk,
    /// sampling, thinking) and MiniMax merges root messages, so a request to
    /// one of their models has to be typed before dispatch. The default and
    /// OpenAI profiles rewrite nothing of their own, so a relay may forward
    /// their requests as the client wrote them.
    pub fn normalizes_chat(self) -> bool {
        self.is_vendor_contract()
    }

    /// Shape the request for dispatch under this profile: the provider's own
    /// normalization first (MiniMax merges the root messages into one
    /// leading root), then every message drops the extension struct that
    /// belongs to another provider, so a foreign field never reaches a
    /// backend or a chat template. Runs from `Normalizable::normalize`, so it
    /// covers every request that enters through `ValidatedJson`. The HTTP
    /// router's streamed pass-through forwards the raw body and skips it,
    /// which is why it hands a chat request to a model whose profile
    /// [`Self::normalizes_chat`] back to the typed path. Only message-level
    /// extension structs are covered; see the module docs. Dropped
    /// extensions are logged once per request. The profile then applies its
    /// contract defaults to fields the client omitted.
    pub fn normalize_chat(self, req: &mut ChatCompletionRequest) {
        match self {
            ProviderProfile::Minimax => minimax::normalize_chat(req),
            ProviderProfile::DeepSeek
            | ProviderProfile::Kimi
            | ProviderProfile::Zai
            | ProviderProfile::OpenAi
            | ProviderProfile::Generic => {}
        }
        let mut dropped: Vec<&'static str> = Vec::new();
        for message in &mut req.messages {
            let role = match message {
                ChatMessage::System { ext, .. } => retain_if(ext, self).then_some("system"),
                ChatMessage::User { ext, .. } => retain_if(ext, self).then_some("user"),
                ChatMessage::Assistant { ext, .. } => retain_if(ext, self).then_some("assistant"),
                ChatMessage::Developer { ext, .. } => retain_if(ext, self).then_some("developer"),
                ChatMessage::Tool { .. }
                | ChatMessage::Function { .. }
                | ChatMessage::Root { .. } => None,
            };
            dropped.extend(role);
        }
        if !dropped.is_empty() {
            // One line per request rather than per message, and the distinct
            // roles rather than one entry per message: the path is client
            // controlled, so both the line count and the line size must be
            // bounded. The model id is what makes a miss diagnosable.
            let count = dropped.len();
            dropped.sort_unstable();
            dropped.dedup();
            tracing::warn!(
                model = %req.model,
                active = ?self,
                dropped = count,
                roles = %dropped.join(","),
                "dropped message extensions that belong to another provider's profile"
            );
        }
        match self {
            ProviderProfile::DeepSeek => deepseek::normalize_chat(req),
            ProviderProfile::Kimi => kimi::normalize_chat(req),
            ProviderProfile::Minimax => minimax::apply_defaults(req),
            ProviderProfile::Zai => zai::normalize_chat(req),
            ProviderProfile::OpenAi | ProviderProfile::Generic => {}
        }
    }

    /// Shape a chat request for an upstream server that renders the prompt
    /// itself, the way the HTTP routers forward it: a field the gateway's own
    /// renderer and the vendor's API read differently goes out as the vendor
    /// reads it. DeepSeek V4.1's `medium` effort is the one such field (the
    /// checkpoint template's level for the gateway's renderer, the vendor's
    /// alias of `high` for an upstream; the table lives with the profile);
    /// every other profile forwards the request [`Self::normalize_chat`]
    /// produced as it is.
    pub fn normalize_chat_for_upstream(self, req: &mut ChatCompletionRequest) {
        match self {
            ProviderProfile::DeepSeek => deepseek::alias_effort_for_upstream(req),
            ProviderProfile::Kimi
            | ProviderProfile::Minimax
            | ProviderProfile::Zai
            | ProviderProfile::OpenAi
            | ProviderProfile::Generic => {}
        }
    }

    /// The call id of a tool result the assistant turn right before it did
    /// not issue, under the contracts that refuse such a history (OpenAI's
    /// and DeepSeek's); `None` otherwise. The rule a history the gateway
    /// assembles (the Responses conversions build one from input items and
    /// stored turns) is held to: every tool result answers a call of the
    /// assistant turn right before it, so a result with no call before it at
    /// all, whether it opens the history or follows another role, answers
    /// nothing and is refused as the public API refuses it, not sent to the
    /// engine as a turn the model never asked for. The gateway's history
    /// loader keeps a stored turn with its calls and drops a stored result
    /// whose call did not survive the reload (a chain cut at a depth cap, at
    /// an expired or deleted record or at another tenant's; a conversation
    /// window), so a result with no call before it is the client's. The chat
    /// endpoint's own history gets the full pairing rule, unanswered calls
    /// included, through [`Self::validate_chat`].
    pub fn unanswerable_tool_result(self, messages: &[ChatMessage]) -> Option<&str> {
        if !matches!(self, ProviderProfile::OpenAi | ProviderProfile::DeepSeek) {
            return None;
        }
        // The call ids of the assistant turn right before the current
        // message: empty when it issued none or another role spoke since.
        let mut issued: Vec<&str> = Vec::new();
        for message in messages {
            match message {
                ChatMessage::Assistant { tool_calls, .. } => {
                    issued = tool_calls
                        .iter()
                        .flatten()
                        .map(|call| call.id.as_str())
                        .collect();
                }
                ChatMessage::Tool { tool_call_id, .. } => {
                    if !issued.contains(&tool_call_id.as_str()) {
                        return Some(tool_call_id.as_str());
                    }
                }
                _ => issued.clear(),
            }
        }
        None
    }

    /// Contract rules applied on top of core validation.
    pub fn validate_chat(
        self,
        req: &ChatCompletionRequest,
    ) -> Result<(), validator::ValidationError> {
        match self {
            ProviderProfile::Kimi => {
                reject_root(req)?;
                kimi::validate_chat(req)
            }
            ProviderProfile::Minimax => minimax::validate_chat(req),
            ProviderProfile::Zai => {
                reject_root(req)?;
                zai::validate_chat(req)
            }
            ProviderProfile::DeepSeek => {
                reject_root(req)?;
                openai::validate_tool_message_pairing(&req.messages)?;
                deepseek::validate_chat(req)
            }
            ProviderProfile::OpenAi => {
                reject_root(req)?;
                openai::validate_chat(req)
            }
            ProviderProfile::Generic => reject_root(req),
        }
    }
}

/// Case-insensitive ASCII prefix test that does not allocate.
fn starts_with_ignore_ascii_case(s: &str, prefix: &str) -> bool {
    s.get(..prefix.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
}

/// Whether the model id names one of OpenAI's own hosted models: `gpt-*`
/// (but not the open-weight `gpt-oss-*`), `chatgpt-*`, `codex-*`, the
/// `o1`/`o3`/`o4` reasoning series, `computer-use-preview`, or a model behind
/// an `openai` route segment such as `openai/gpt-4.1`.
///
/// The rules the public OpenAI API enforces on request *content* beyond the
/// schema (a strict-mode schema must pin `additionalProperties: false` on
/// every object node; a `json_object` format needs the word "json" in the
/// prompt) hold for those models only. A self-hosted model passes such
/// requests through to the engine, whose grammar compiler decides what it can
/// constrain, so the gateway does not refuse ahead of it.
pub fn is_openai_vendor_model(model: &str) -> bool {
    const PREFIXES: [&str; 6] = [
        "gpt-",
        "chatgpt-",
        "codex-",
        "computer-use-preview",
        "text-embedding-",
        "omni-moderation",
    ];
    // The open-weight gpt-oss models are served locally whatever route
    // segment precedes them: never the vendor's hosted contract.
    if model
        .split('/')
        .any(|segment| starts_with_ignore_ascii_case(segment, "gpt-oss"))
    {
        return false;
    }
    model.split('/').any(|segment| {
        segment.eq_ignore_ascii_case("openai")
            || PREFIXES
                .iter()
                .any(|prefix| starts_with_ignore_ascii_case(segment, prefix))
            || is_openai_reasoning_series(segment)
    })
}

/// `o1`, `o3`, `o4-mini`, `o3-pro-2025-06-10`: an `o`, the series digit, then
/// the end of the segment or a dash.
fn is_openai_reasoning_series(segment: &str) -> bool {
    let mut chars = segment.chars();
    matches!(chars.next(), Some('o' | 'O'))
        && matches!(chars.next(), Some('1' | '3' | '4'))
        && matches!(chars.next(), None | Some('-'))
}

/// The `root` role is a MiniMax-only extension; other dialects reject it the
/// way their reference APIs do.
fn reject_root(req: &ChatCompletionRequest) -> Result<(), validator::ValidationError> {
    if req
        .messages
        .iter()
        .any(|m| matches!(m, ChatMessage::Root { .. }))
    {
        return Err(invalid("invalid_role", "invalid role: root"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `normalizes_chat` is the body-blind form of `normalize_chat`: a profile
    /// claims to shape requests exactly when its normalization changes a
    /// plain streamed request the client sent without the fields the profile
    /// fills, and the two profiles that claim nothing leave it as sent.
    #[test]
    fn normalizes_chat_agrees_with_what_normalize_chat_does() {
        let cases = [
            ("kimi-k2", ProviderProfile::Kimi),
            ("MiniMax-M3", ProviderProfile::Minimax),
            ("glm-5", ProviderProfile::Zai),
            ("deepseek-v4-pro", ProviderProfile::DeepSeek),
            ("gpt-4o", ProviderProfile::OpenAi),
            ("self-hosted-model", ProviderProfile::Generic),
        ];
        for (model, expected) in cases {
            let profile = ProviderProfile::for_model(model);
            assert_eq!(profile, expected, "{model}");
            let mut request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
                "model": model,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": true,
            }))
            .expect("request deserializes");
            let as_sent = serde_json::to_value(&request).expect("request serializes");
            profile.normalize_chat(&mut request);
            let shaped = serde_json::to_value(&request).expect("request serializes");
            assert_eq!(
                shaped != as_sent,
                profile.normalizes_chat(),
                "{model}: normalize_chat changed the request ({}) but normalizes_chat says {}",
                shaped != as_sent,
                profile.normalizes_chat()
            );
        }
    }

    #[test]
    fn only_the_kimi_profile_exposes_message_declared_tools() {
        let request = |model: &str| -> ChatCompletionRequest {
            serde_json::from_value(serde_json::json!({
                "model": model,
                "messages": [
                    {"role": "system", "content": "", "tools": [
                        {"type": "function", "function": {"name": "get_weather"}}
                    ]},
                    {"role": "developer", "content": "", "tools": [
                        {"type": "function", "function": {"name": "get_time"}}
                    ]},
                    {"role": "user", "content": "hi"}
                ]
            }))
            .expect("request deserializes")
        };

        let kimi = request("kimi-k3");
        let names: Vec<&str> = ProviderProfile::for_model(&kimi.model)
            .dynamic_tools(&kimi)
            .map(|tool| tool.function.name.as_str())
            .collect();
        assert_eq!(names, ["get_weather", "get_time"]);

        for model in ["gpt-4o", "MiniMax-M2"] {
            let other = request(model);
            assert_eq!(
                ProviderProfile::for_model(&other.model)
                    .dynamic_tools(&other)
                    .count(),
                0,
                "{model} must not pick up message-declared tools"
            );
        }
    }

    #[test]
    fn a_vendor_named_alias_gives_the_served_model_its_vendor_profile() {
        // Names unique to this test: the alias table is process-wide.
        ProviderProfile::register_model_aliases([
            ("kimi-k3-public-name", "served-model-alpha"),
            ("gpt-4o-public-name", "served-model-beta"),
            // A second, disagreeing alias of the same served model is ignored.
            ("MiniMax-M3-public-name", "served-model-alpha"),
        ]);
        assert_eq!(
            ProviderProfile::for_model("served-model-alpha"),
            ProviderProfile::Kimi
        );
        // An OpenAI-looking alias is compatibility naming, not a contract:
        // the served model stays on the default profile, and a request
        // naming the alias is judged as that model, not under OpenAI's rules.
        assert_eq!(
            ProviderProfile::for_model("served-model-beta"),
            ProviderProfile::Generic
        );
        assert_eq!(
            ProviderProfile::for_model("gpt-4o-public-name"),
            ProviderProfile::Generic
        );
        assert_eq!(
            ProviderProfile::for_model("kimi-k3-public-name"),
            ProviderProfile::Kimi
        );
        assert_eq!(
            ProviderProfile::for_model("served-model-gamma"),
            ProviderProfile::Generic
        );

        // A request that entered under the vendor name and had its model id
        // rewritten to the served name keeps the profile's dynamic tools.
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "served-model-alpha",
            "messages": [
                {"role": "system", "content": "", "tools": [
                    {"type": "function", "function": {"name": "get_weather"}}
                ]},
                {"role": "user", "content": "hi"}
            ],
            "tool_choice": "required"
        }))
        .expect("request deserializes");
        let names: Vec<&str> = request
            .effective_tools()
            .map(|tool| tool.function.name.as_str())
            .collect();
        assert_eq!(names, ["get_weather"]);
        assert_eq!(request.callable_tools().len(), 1);
    }

    #[test]
    fn a_registered_name_is_read_through_the_names_that_carry_its_contract() {
        // Names unique to this test: the alias table is process-wide.
        ProviderProfile::register_model_aliases([
            (
                "opaque-alias-of-a-vendor-named-model",
                "kimi-k3-vendor-named-served-model",
            ),
            (
                "MiniMax-M3-first-public-name",
                "opaque-served-model-under-vendor-names",
            ),
            (
                "MiniMax-M3-second-public-name",
                "opaque-served-model-under-vendor-names",
            ),
            (
                "opaque-alias-of-the-opaque-served-model",
                "opaque-served-model-under-vendor-names",
            ),
        ]);
        // An opaque alias of a vendor-named served model is read as that model.
        assert!(served_under(
            "opaque-alias-of-a-vendor-named-model",
            |name| name == "kimi-k3-vendor-named-served-model"
        ));
        assert!(!served_under(
            "opaque-alias-of-a-vendor-named-model",
            |name| name == "opaque-alias-of-a-vendor-named-model"
        ));
        // An opaque served model, and an opaque alias of it, are read as the
        // vendor names it is served under, which have to agree; a
        // vendor-named alias is read as its own name.
        for model in [
            "opaque-served-model-under-vendor-names",
            "opaque-alias-of-the-opaque-served-model",
        ] {
            assert_eq!(ProviderProfile::for_model(model), ProviderProfile::Minimax);
            assert!(
                served_under(model, |name| name.starts_with("MiniMax-M3-")),
                "{model}"
            );
            assert!(
                !served_under(model, |name| name == "MiniMax-M3-first-public-name"),
                "{model}: the second public name disagrees"
            );
        }
        for model in [
            "MiniMax-M3-first-public-name",
            "MiniMax-M3-second-public-name",
        ] {
            assert_eq!(ProviderProfile::for_model(model), ProviderProfile::Minimax);
            assert!(served_under(model, |name| name == model), "{model}");
        }
        // One vendor's name for another vendor's model is read as that model.
        ProviderProfile::register_model_aliases([(
            "kimi-k2-name-of-a-minimax-model",
            "MiniMax-M3-served-under-a-kimi-name",
        )]);
        assert_eq!(
            ProviderProfile::for_model("kimi-k2-name-of-a-minimax-model"),
            ProviderProfile::Minimax
        );
        assert!(served_under("kimi-k2-name-of-a-minimax-model", |name| name
            == "MiniMax-M3-served-under-a-kimi-name"));
        // A name the table does not know is read as itself.
        assert!(served_under("unregistered-model", |name| name == "unregistered-model"));
    }

    #[test]
    fn only_the_minimax_profile_parses_tool_calls_without_tools() {
        assert!(ProviderProfile::Minimax.parses_tool_calls_without_tools());
        assert!(!ProviderProfile::Kimi.parses_tool_calls_without_tools());
        assert!(!ProviderProfile::Zai.parses_tool_calls_without_tools());
        assert!(!ProviderProfile::OpenAi.parses_tool_calls_without_tools());
        assert!(!ProviderProfile::Generic.parses_tool_calls_without_tools());
    }

    #[test]
    fn openai_vendor_models_are_told_apart_from_self_hosted_ones() {
        for model in [
            "gpt-4.1",
            "gpt-5-nano",
            "GPT-4o-mini",
            "chatgpt-4o-latest",
            "o1",
            "o3-pro-2025-06-10",
            "o4-mini",
            "codex-mini-latest",
            "computer-use-preview",
            "openai/gpt-4.1",
            "openrouter/openai/o3",
        ] {
            assert!(is_openai_vendor_model(model), "{model}");
        }
        for model in [
            "Qwen/Qwen2.5-14B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "openai/gpt-oss-120b",
            "gpt-oss-20b",
            "olmo-2",
            "o5",
            "m1",
            "",
        ] {
            assert!(!is_openai_vendor_model(model), "{model}");
        }
    }

    #[test]
    fn model_id_selects_profile() {
        for model in [
            "kimi-k3",
            "Kimi-K2.6",
            "/models/Kimi-K3",
            "moonshotai/kimi-k2",
            "openrouter/moonshotai/kimi-k2",
            "MoonshotAI/Kimi-K2-Instruct",
        ] {
            assert_eq!(
                ProviderProfile::for_model(model),
                ProviderProfile::Kimi,
                "{model}"
            );
        }
        for model in [
            "MiniMax-M3",
            "/models/MiniMax-M2",
            "MiniMaxAI/MiniMax-M2",
            "abab6.5s-chat",
        ] {
            assert_eq!(
                ProviderProfile::for_model(model),
                ProviderProfile::Minimax,
                "{model}"
            );
        }
        for model in [
            "glm-5.3-flash",
            "GLM-5.3-Flash",
            "zai-org/GLM-5.3-Flash",
            "/models/glm-4.7",
            "z-ai/glm-5",
        ] {
            assert_eq!(
                ProviderProfile::for_model(model),
                ProviderProfile::Zai,
                "{model}"
            );
        }
        for model in [
            "gpt-4o-mini",
            "GPT-5-nano",
            "openai/gpt-4o",
            "openrouter/openai/o3-mini",
            "chatgpt-4o-latest",
            "codex-mini-latest",
            "o1",
            "o4-mini-2025-04-16",
        ] {
            assert_eq!(
                ProviderProfile::for_model(model),
                ProviderProfile::OpenAi,
                "{model}"
            );
        }
        for model in [
            "",
            "/models/llama-3",
            "qwen3-8b",
            "Qwen/Qwen3-VL-8B-Instruct",
            "gpt-oss-20b",
            "openai/gpt-oss-120b",
            "olmo-3-7b-instruct",
            "o1js-chat",
            "my-kimi-alias",
            "my-glm-alias",
            // ChatGLM predates the z.ai chat contract.
            "THUDM/chatglm3-6b",
        ] {
            assert_eq!(
                ProviderProfile::for_model(model),
                ProviderProfile::Generic,
                "{model}"
            );
        }
    }

    /// A forced tool choice with thinking on is the engine's to serve for a
    /// self-hosted DeepSeek V4 or V4.1 model (the gateway shapes the call's
    /// grammar for where the engine starts it); the vendor's own API refuses
    /// it by itself where it still does. The profile's other rules stay.
    #[test]
    fn a_forced_tool_choice_with_thinking_on_is_not_refused_for_deepseek_v4() {
        let request = |model: &str,
                       tool_choice: serde_json::Value,
                       extra: serde_json::Value|
         -> ChatCompletionRequest {
            let mut body = serde_json::json!({
                "model": model,
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
                "tools": [{"type": "function", "function": {"name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
                "tool_choice": tool_choice,
            });
            body.as_object_mut()
                .expect("object")
                .extend(extra.as_object().cloned().unwrap_or_default());
            serde_json::from_value(body).expect("request deserializes")
        };
        let code = |profile: ProviderProfile, req: &ChatCompletionRequest| {
            profile
                .validate_chat(req)
                .err()
                .map(|error| error.code.to_string())
        };
        let named = serde_json::json!({"type": "function", "function": {"name": "get_weather"}});
        let thinking_on = [
            serde_json::json!({}),
            serde_json::json!({"thinking": {"type": "enabled"}}),
            serde_json::json!({"chat_template_kwargs": {"thinking": true}}),
            serde_json::json!({"reasoning_effort": "high"}),
        ];
        for model in [
            "deepseek-v4.1-flash",
            "deepseek-ai/DeepSeek-V4.1-Flash",
            "deepseek-flash",
            "deepseek-v4-pro",
        ] {
            let profile = ProviderProfile::for_model(model);
            assert_eq!(profile, ProviderProfile::DeepSeek, "{model}");
            for choice in [named.clone(), serde_json::json!("required")] {
                for extra in &thinking_on {
                    let mut req = request(model, choice.clone(), extra.clone());
                    profile.normalize_chat(&mut req);
                    assert_eq!(code(profile, &req), None, "{model} {choice} {extra}");
                }
            }
        }
        // The adaptive thinking type is still no DeepSeek V4 setting.
        let mut req = request(
            "deepseek-v4.1-flash",
            named,
            serde_json::json!({"thinking": {"type": "adaptive"}}),
        );
        ProviderProfile::DeepSeek.normalize_chat(&mut req);
        assert_eq!(
            code(ProviderProfile::DeepSeek, &req).as_deref(),
            Some("thinking_type_not_supported")
        );
    }

    /// A tool result no call of the assistant turn right before it issued is
    /// a malformed history under the DeepSeek contract, as under OpenAI's:
    /// the chat endpoint refuses it with the public API's code, and a history
    /// assembled elsewhere names the id, for an unknown id as for a result
    /// with no call before it at all; the generic profile leaves every shape
    /// to the engine.
    #[test]
    fn a_tool_result_without_its_call_is_refused_under_the_deepseek_contract() {
        let request = |model: &str, messages: serde_json::Value| -> ChatCompletionRequest {
            serde_json::from_value(serde_json::json!({"model": model, "messages": messages}))
                .expect("a chat request")
        };
        let wrong = |model: &str| {
            request(
                model,
                serde_json::json!([
                    {"role": "user", "content": "Add 17 and 25."},
                    {"role": "assistant", "content": null, "tool_calls": [{
                        "id": "call_1", "type": "function",
                        "function": {"name": "add_integers", "arguments": "{\"a\":17,\"b\":25}"}
                    }]},
                    {"role": "tool", "tool_call_id": "call_doesnotexist", "content": "42"}
                ]),
            )
        };
        let right = request(
            "deepseek-flash",
            serde_json::json!([
                {"role": "user", "content": "Add 17 and 25."},
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "add_integers", "arguments": "{\"a\":17,\"b\":25}"}
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "42"}
            ]),
        );
        let code = |request: &ChatCompletionRequest| {
            ProviderProfile::for_model(&request.model)
                .validate_chat(request)
                .err()
                .map(|error| error.code.to_string())
        };
        assert_eq!(
            code(&wrong("deepseek-flash")).as_deref(),
            Some("tool_call_id_not_found")
        );
        assert_eq!(code(&right), None);
        assert_eq!(code(&wrong("local-model")), None);

        let deepseek = ProviderProfile::DeepSeek;
        assert_eq!(
            deepseek.unanswerable_tool_result(&wrong("deepseek-flash").messages),
            Some("call_doesnotexist")
        );
        assert_eq!(deepseek.unanswerable_tool_result(&right.messages), None);
        assert_eq!(
            ProviderProfile::Generic.unanswerable_tool_result(&wrong("local-model").messages),
            None
        );
        // The shapes a Responses history takes, under the same rule. A result
        // for one of two calls answers a call the turn before it issued.
        let two_calls_one_result = request(
            "deepseek-flash",
            serde_json::json!([
                {"role": "user", "content": "Add 17 and 25, then 3 and 4."},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_a", "type": "function",
                     "function": {"name": "add_integers", "arguments": "{\"a\":17,\"b\":25}"}},
                    {"id": "call_b", "type": "function",
                     "function": {"name": "add_integers", "arguments": "{\"a\":3,\"b\":4}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_b", "content": "7"}
            ]),
        );
        assert_eq!(
            deepseek.unanswerable_tool_result(&two_calls_one_result.messages),
            None
        );
        // A result with no call before it answers nothing: after the user's
        // message, as the first message ahead of the call it names, and after
        // a user turn that ended the turn it could have answered.
        let orphan = request(
            "deepseek-flash",
            serde_json::json!([
                {"role": "user", "content": "Add 17 and 25."},
                {"role": "tool", "tool_call_id": "call_a", "content": "42"}
            ]),
        );
        assert_eq!(
            deepseek.unanswerable_tool_result(&orphan.messages),
            Some("call_a")
        );
        let result_before_call = request(
            "deepseek-flash",
            serde_json::json!([
                {"role": "tool", "tool_call_id": "call_a", "content": "42"},
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_a", "type": "function",
                    "function": {"name": "add_integers", "arguments": "{\"a\":17,\"b\":25}"}
                }]}
            ]),
        );
        assert_eq!(
            deepseek.unanswerable_tool_result(&result_before_call.messages),
            Some("call_a")
        );
        let user_between = request(
            "deepseek-flash",
            serde_json::json!([
                {"role": "user", "content": "Add 17 and 25."},
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_a", "type": "function",
                    "function": {"name": "add_integers", "arguments": "{\"a\":17,\"b\":25}"}
                }]},
                {"role": "user", "content": "Never mind."},
                {"role": "tool", "tool_call_id": "call_a", "content": "42"}
            ]),
        );
        assert_eq!(
            deepseek.unanswerable_tool_result(&user_between.messages),
            Some("call_a")
        );
        assert_eq!(
            ProviderProfile::Generic.unanswerable_tool_result(&orphan.messages),
            None
        );
    }
}
