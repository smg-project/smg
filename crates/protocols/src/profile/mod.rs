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
    sync::{OnceLock, RwLock},
};

use crate::{
    chat::{ChatCompletionRequest, ChatMessage},
    common::Tool,
    ext::retain_if,
};

/// Served model ids that take a vendor's profile from an alias they are served
/// under, recorded once at startup from the configured model aliases. An alias
/// is resolved into the request's model id before the response side reads it,
/// so the served name has to select the profile its vendor-named alias does,
/// or a request that entered under the vendor name loses the profile half-way.
fn alias_profiles() -> &'static RwLock<HashMap<String, ProviderProfile>> {
    static ALIAS_PROFILES: OnceLock<RwLock<HashMap<String, ProviderProfile>>> = OnceLock::new();
    ALIAS_PROFILES.get_or_init(|| RwLock::new(HashMap::new()))
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
        let registered = alias_profiles()
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(model)
            .copied();
        registered.unwrap_or_else(|| Self::from_model_segments(model))
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
    pub fn register_model_aliases<'a>(aliases: impl IntoIterator<Item = (&'a str, &'a str)>) {
        let aliases: Vec<(&str, &str)> = aliases.into_iter().collect();
        let mut map = alias_profiles().write().unwrap_or_else(|e| e.into_inner());
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
            match map.get(*canonical) {
                Some(existing) if *existing != profile => tracing::warn!(
                    canonical,
                    alias,
                    kept = ?existing,
                    ignored = ?profile,
                    "model alias names a different vendor than an earlier alias of the same served model"
                ),
                Some(_) => {}
                None => {
                    map.insert((*canonical).to_string(), profile);
                }
            }
        }
        // ... then every alias resolves to its served model's profile.
        for (alias, canonical) in &aliases {
            let profile = map
                .get(*canonical)
                .copied()
                .unwrap_or_else(|| Self::from_model_segments(canonical));
            map.insert((*alias).to_string(), profile);
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

    /// Shape the request for dispatch under this profile: the provider's own
    /// normalization first (MiniMax folds every root message into a
    /// leading system message), then every message drops the extension struct that
    /// belongs to another provider, so a foreign field never reaches a
    /// backend or a chat template. Runs from `Normalizable::normalize`, so it
    /// covers every request that enters through `ValidatedJson`; the HTTP
    /// router's streamed pass-through forwards the raw body and skips it.
    /// Only message-level extension structs
    /// are covered; see the module docs. Dropped extensions are logged once
    /// per request. The profile then applies its contract defaults to fields
    /// the client omitted.
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
            ProviderProfile::Zai => zai::normalize_chat(req),
            ProviderProfile::OpenAi | ProviderProfile::Generic | ProviderProfile::Minimax => {}
        }
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
        let mut e = validator::ValidationError::new("invalid_role");
        e.message = Some("invalid role: root".into());
        return Err(e);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
