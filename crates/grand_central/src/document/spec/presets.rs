//! The presets of [`PolicySpec`]: their names, the document each one makes
//! of its policy's profile, and the documents a pair and an
//! encode-prefill-decode plan compose of them.

use std::collections::BTreeMap;

use super::{
    BindingSpec, ExplainSpec, KeySourceSpec, PlanName, PlansSpec, PolicySpec, RoutingKeySpec,
    SpecError, ViewSpec, API_VERSION,
};
use crate::policies;

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
}
