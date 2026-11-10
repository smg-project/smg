//! `PolicySpec` (design 4.6): the one versioned document. Profiles name
//! stages, bindings match requests to profiles or plans, and validation
//! answers what the host must feed before any request exists. The document
//! has an `apiVersion` and no `kind`: its forms are a file and the API. An
//! unknown field is refused by name, never ignored.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{
    builtins::{
        Best, HasCapacity, HasRoom, Healthy, LeastLoad, LeastRequests, NotOverloaded, NotTried,
        Passthrough, PowerOfTwo, PrefixAffinity, Random, RoundRobin, SessionAffinity,
    },
    framework::{Binding, Engine, Filter, Picker, Plan, Profile, Scorer, Weighted},
    view::Phase,
};

pub const API_VERSION: &str = "smg.policy/v1";

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
}

impl Needs {
    fn or(self, other: Self) -> Self {
        Self {
            text: self.text || other.text,
            token_hashes: self.token_hashes || other.token_hashes,
            text_hashes: self.text_hashes || other.text_hashes,
            loads: self.loads || other.loads,
            events: self.events || other.events,
            ring: self.ring || other.ring,
            headers: self.headers || other.headers,
            media: self.media || other.media,
        }
    }
}

/// A scorer reference: a bare name, or a name with its weight and parameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ScoreSpec {
    Name(String),
    Named(BTreeMap<String, ScoreParams>),
}

/// A scorer's weight and the parameters of the least-load scorer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreParams {
    #[serde(default = "one")]
    pub weight: f32,
    #[serde(default)]
    pub kv_pressure_weight: Option<f32>,
    #[serde(default)]
    pub mean_prefill_tokens: Option<u32>,
    #[serde(default)]
    pub default_throughput: Option<f32>,
}

fn one() -> f32 {
    1.0
}

impl ScoreSpec {
    fn entries(&self) -> Vec<(&str, ScoreParams)> {
        match self {
            Self::Name(name) => vec![(
                name.as_str(),
                ScoreParams {
                    weight: 1.0,
                    kv_pressure_weight: None,
                    mean_prefill_tokens: None,
                    default_throughput: None,
                },
            )],
            Self::Named(map) => map.iter().map(|(k, v)| (k.as_str(), v.clone())).collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileSpec {
    #[serde(default)]
    pub filter: Vec<String>,
    #[serde(default)]
    pub score: Vec<ScoreSpec>,
    pub pick: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Match {
    #[serde(default = "any_model")]
    pub model: String,
    #[serde(default)]
    pub phase: Vec<Phase>,
}

impl Default for Match {
    fn default() -> Self {
        Self {
            model: any_model(),
            phase: Vec::new(),
        }
    }
}

fn any_model() -> String {
    "*".to_string()
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BindingSpec {
    #[serde(rename = "match", default)]
    pub matcher: Match,
    /// The profile of a single decision.
    #[serde(default)]
    pub profile: Option<String>,
    /// `single` (the default) or `pd-pair`.
    #[serde(default)]
    pub plan: Option<String>,
    /// The profiles per side: `single`, or `prefill` and `decode` under the
    /// `pd-pair` plan, both named.
    #[serde(default)]
    pub profiles: BTreeMap<String, String>,
}

/// The spec document.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicySpec {
    #[serde(rename = "apiVersion")]
    pub api_version: String,
    #[serde(default)]
    pub generation: u64,
    pub profiles: BTreeMap<String, ProfileSpec>,
    #[serde(default)]
    pub bindings: Vec<BindingSpec>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SpecError {
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
    #[error("binding {index}: unknown plan {name:?}")]
    UnknownPlan { index: usize, name: String },
    #[error("binding {index} names no profile")]
    NoProfile { index: usize },
    #[error("binding {index}: the pd-pair plan names no {side} profile")]
    PairSideUnnamed { index: usize, side: &'static str },
    #[error("binding {index}: the pd-pair plan takes its profiles per side, not one profile")]
    ProfileUnderPair { index: usize },
    #[error("binding {index}: profiles names an unknown side {side:?}")]
    UnknownSide { index: usize, side: String },
    #[error(
        "binding {index}: the pd-pair plan's phase list leaves out {side}; name both sides or none"
    )]
    PairPhases { index: usize, side: &'static str },
    #[error("binding {index}: model {model:?} is not an id, `prefix*`, `*suffix` or `*`")]
    BadModelGlob { index: usize, model: String },
    #[error("profile {profile:?}: scorer {scorer:?} parameter {name} = {value} is not {reason}")]
    BadParameter {
        profile: String,
        scorer: String,
        name: &'static str,
        value: String,
        reason: &'static str,
    },
    #[error("profile {profile:?}: scorer {scorer:?} takes no parameter {name}")]
    ParameterOutOfPlace {
        profile: String,
        scorer: String,
        name: &'static str,
    },
    #[error("no profiles")]
    Empty,
    #[error("unknown preset {0:?}")]
    UnknownPreset(String),
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

/// A scorer's parameters, refused by name: the weight is finite and above
/// zero; the least-load parameters are finite, the KV-pressure weight at or
/// above zero and the other two above it; no other scorer takes them.
fn check_params(profile: &str, scorer: &str, p: &ScoreParams) -> Result<(), SpecError> {
    let bad = |name: &'static str, value: String, reason: &'static str| SpecError::BadParameter {
        profile: profile.to_string(),
        scorer: scorer.to_string(),
        name,
        value,
        reason,
    };
    if !(p.weight.is_finite() && p.weight > 0.0) {
        return Err(bad(
            "weight",
            p.weight.to_string(),
            "a finite number above zero",
        ));
    }
    if scorer != "least-load" {
        let present = [
            ("kv_pressure_weight", p.kv_pressure_weight.is_some()),
            ("mean_prefill_tokens", p.mean_prefill_tokens.is_some()),
            ("default_throughput", p.default_throughput.is_some()),
        ];
        if let Some(&(name, _)) = present.iter().find(|&&(_, given)| given) {
            return Err(SpecError::ParameterOutOfPlace {
                profile: profile.to_string(),
                scorer: scorer.to_string(),
                name,
            });
        }
        return Ok(());
    }
    if let Some(v) = p
        .kv_pressure_weight
        .filter(|v| !(v.is_finite() && *v >= 0.0))
    {
        return Err(bad(
            "kv_pressure_weight",
            v.to_string(),
            "a finite number at or above zero",
        ));
    }
    if p.mean_prefill_tokens == Some(0) {
        return Err(bad("mean_prefill_tokens", "0".to_string(), "above zero"));
    }
    if let Some(v) = p
        .default_throughput
        .filter(|v| !(v.is_finite() && *v > 0.0))
    {
        return Err(bad(
            "default_throughput",
            v.to_string(),
            "a finite number above zero",
        ));
    }
    Ok(())
}

fn scorer_by_name(name: &str, params: &ScoreParams) -> Option<Box<dyn Scorer>> {
    Some(match name {
        "least-load" => Box::new(LeastLoad::new(
            params.kv_pressure_weight.unwrap_or(0.15),
            params.mean_prefill_tokens.unwrap_or(1024),
            params.default_throughput.unwrap_or(2000.0),
        )),
        "least-requests" => Box::new(LeastRequests),
        "session-affinity" => Box::new(SessionAffinity),
        "prefix-affinity" => Box::new(PrefixAffinity),
        _ => return None,
    })
}

fn picker_by_name(name: &str) -> Option<Box<dyn Picker>> {
    Some(match name {
        "best" => Box::new(Best),
        "round-robin" => Box::new(RoundRobin::new()),
        "random" => Box::new(Random),
        "power-of-two" => Box::new(PowerOfTwo),
        "passthrough" => Box::new(Passthrough),
        _ => return None,
    })
}

/// A sampling picker compares its draws by the profile's scorers, so it
/// needs at least one.
const SAMPLING_PICKERS: [&str; 1] = ["power-of-two"];

/// The ten preset names of today, as whole profiles. The ones whose stages
/// this build does not carry yet answer `UnknownPreset`.
pub const PRESETS: [&str; 5] = [
    "random",
    "round_robin",
    "passthrough",
    "power_of_two",
    "least_load",
];

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
                score: vec![ScoreSpec::Name("least-load".into())],
                pick: "power-of-two".into(),
            },
            "least_load" => ProfileSpec {
                filter: standard_filters(),
                score: vec![ScoreSpec::Name("least-load".into())],
                pick: "best".into(),
            },
            other => return Err(SpecError::UnknownPreset(other.to_string())),
        };
        let mut profiles = BTreeMap::new();
        profiles.insert(name.to_string(), profile);
        Ok(Self {
            api_version: API_VERSION.to_string(),
            generation: 1,
            profiles,
            bindings: vec![BindingSpec {
                profile: Some(name.to_string()),
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
            plan: Some("pd-pair".into()),
            profiles,
            ..BindingSpec::default()
        }];
        Ok(spec)
    }

    /// Validate the document and build the engine; the needs summary comes
    /// with it.
    pub fn validate(&self) -> Result<Engine, SpecError> {
        if self.api_version != API_VERSION {
            return Err(SpecError::ApiVersion(self.api_version.clone()));
        }
        if self.profiles.is_empty() {
            return Err(SpecError::Empty);
        }
        let mut profiles = Vec::with_capacity(self.profiles.len());
        let mut index: BTreeMap<&str, u16> = BTreeMap::new();
        let mut needs = Needs::default();
        for (name, p) in &self.profiles {
            let mut filters = Vec::with_capacity(p.filter.len());
            for f in &p.filter {
                filters.push(filter_by_name(f).ok_or_else(|| SpecError::UnknownFilter {
                    profile: name.clone(),
                    name: f.clone(),
                })?);
            }
            let mut scorers = Vec::new();
            for s in &p.score {
                for (scorer_name, params) in s.entries() {
                    // The name first, so a misspelt scorer with parameters is
                    // refused as unknown and not for its parameters.
                    let scorer = scorer_by_name(scorer_name, &params).ok_or_else(|| {
                        SpecError::UnknownScorer {
                            profile: name.clone(),
                            name: scorer_name.to_string(),
                        }
                    })?;
                    check_params(name, scorer_name, &params)?;
                    needs = needs.or(scorer.needs());
                    scorers.push(Weighted {
                        scorer,
                        weight: params.weight,
                    });
                }
            }
            let picker = picker_by_name(&p.pick).ok_or_else(|| SpecError::UnknownPicker {
                profile: name.clone(),
                name: p.pick.clone(),
            })?;
            if scorers.is_empty() && SAMPLING_PICKERS.contains(&p.pick.as_str()) {
                return Err(SpecError::PickerNeedsScorer {
                    profile: name.clone(),
                    picker: p.pick.clone(),
                });
            }
            index.insert(name.as_str(), profiles.len() as u16);
            profiles.push(Profile {
                name: name.clone(),
                filters,
                scorers,
                picker,
            });
        }
        let mut bindings = Vec::with_capacity(self.bindings.len());
        for (i, b) in self.bindings.iter().enumerate() {
            let resolve = |name: &str| {
                index
                    .get(name)
                    .copied()
                    .ok_or_else(|| SpecError::UnknownProfile {
                        index: i,
                        name: name.to_string(),
                    })
            };
            let side = |name: &str| b.profiles.get(name).map(String::as_str);
            if let Some(unknown) = b
                .profiles
                .keys()
                .find(|k| !matches!(k.as_str(), "single" | "prefill" | "decode"))
            {
                return Err(SpecError::UnknownSide {
                    index: i,
                    side: unknown.clone(),
                });
            }
            // The matcher understands an id, `prefix*`, `*suffix` and `*`;
            // any other star would match only its literal self.
            let model = b.matcher.model.as_str();
            let stars = model.matches('*').count();
            let glob_ok = !model.is_empty()
                && (stars == 0 || (stars == 1 && (model.starts_with('*') || model.ends_with('*'))));
            if !glob_ok {
                return Err(SpecError::BadModelGlob {
                    index: i,
                    model: model.to_string(),
                });
            }
            let (plan, profile, decode_profile) = match b.plan.as_deref() {
                None | Some("single") => {
                    let name = b
                        .profile
                        .as_deref()
                        .or_else(|| side("single"))
                        .ok_or(SpecError::NoProfile { index: i })?;
                    (Plan::Single, resolve(name)?, None)
                }
                Some("pd-pair") => {
                    if b.profile.is_some() {
                        return Err(SpecError::ProfileUnderPair { index: i });
                    }
                    let prefill = side("prefill").ok_or(SpecError::PairSideUnnamed {
                        index: i,
                        side: "prefill",
                    })?;
                    let decode = side("decode").ok_or(SpecError::PairSideUnnamed {
                        index: i,
                        side: "decode",
                    })?;
                    // A pair resolves from the prefill side's lookup and
                    // serves both sides, so its phase list names both or none.
                    if !b.matcher.phase.is_empty() {
                        for (phase, side) in
                            [(Phase::Prefill, "prefill"), (Phase::Decode, "decode")]
                        {
                            if !b.matcher.phase.contains(&phase) {
                                return Err(SpecError::PairPhases { index: i, side });
                            }
                        }
                    }
                    (Plan::PdPair, resolve(prefill)?, Some(resolve(decode)?))
                }
                Some(other) => {
                    return Err(SpecError::UnknownPlan {
                        index: i,
                        name: other.to_string(),
                    })
                }
            };
            bindings.push(Binding {
                model: b.matcher.model.clone(),
                phases: b.matcher.phase.clone(),
                plan,
                profile,
                decode_profile,
            });
        }
        if bindings.is_empty() {
            bindings.push(Binding {
                model: "*".into(),
                phases: vec![],
                plan: Plan::Single,
                profile: 0,
                decode_profile: None,
            });
        }
        Ok(Engine::new(profiles, bindings, self.generation, needs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::Phase;

    #[test]
    fn every_preset_validates_and_declares_its_needs() {
        for name in PRESETS {
            let spec = PolicySpec::preset(name).expect(name);
            let engine = spec.validate().expect(name);
            let load_aware = name == "least_load" || name == "power_of_two";
            assert_eq!(engine.needs().loads, load_aware, "{name}");
            assert!(!engine.needs().events, "{name}");
            assert!(engine.profile_for("any", Phase::Single).is_some(), "{name}");
        }
        assert_eq!(
            PolicySpec::preset("cache_aware").err(),
            Some(SpecError::UnknownPreset("cache_aware".into()))
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
        let spec: PolicySpec = serde_json::from_str(text).expect("parses");
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
        bad.profiles.get_mut("default").expect("default").pick = "manual".into();
        assert_eq!(
            bad.validate().err(),
            Some(SpecError::UnknownPicker {
                profile: "default".into(),
                name: "manual".into()
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
    fn a_kind_field_or_any_unknown_field_is_refused_by_name() {
        let with_kind = r#"{ "apiVersion": "smg.policy/v1", "kind": "PolicySpec",
          "profiles": { "p": { "pick": "random" } } }"#;
        let err = serde_json::from_str::<PolicySpec>(with_kind)
            .expect_err("a kind field is refused")
            .to_string();
        assert!(err.contains("unknown field `kind`"), "{err}");
        let with_view = r#"{ "apiVersion": "smg.policy/v1", "view": { "headers": [] },
          "profiles": { "p": { "pick": "random" } } }"#;
        let err = serde_json::from_str::<PolicySpec>(with_view)
            .expect_err("an undeclared section is refused")
            .to_string();
        assert!(err.contains("unknown field `view`"), "{err}");
        let with_sessions = r#"{ "apiVersion": "smg.policy/v1",
          "profiles": { "p": { "pick": "random" } },
          "bindings": [ { "profile": "p", "sessions": { "ttl_s": 600 } } ] }"#;
        let err = serde_json::from_str::<PolicySpec>(with_sessions)
            .expect_err("an undeclared binding field is refused")
            .to_string();
        assert!(err.contains("unknown field `sessions`"), "{err}");
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
    fn a_pair_binding_names_both_sides_or_is_refused() {
        let mut spec = PolicySpec::pair_presets("least_load", "least_load").expect("pair");
        spec.bindings[0].profiles.remove("decode");
        assert_eq!(
            spec.validate().err(),
            Some(SpecError::PairSideUnnamed {
                index: 0,
                side: "decode"
            })
        );
        let mut one_profile = PolicySpec::pair_presets("least_load", "least_load").expect("pair");
        one_profile.bindings[0].profile = Some("prefill".into());
        assert_eq!(
            one_profile.validate().err(),
            Some(SpecError::ProfileUnderPair { index: 0 })
        );
        let mut odd_side = PolicySpec::pair_presets("least_load", "least_load").expect("pair");
        odd_side.bindings[0]
            .profiles
            .insert("encode".into(), "prefill".into());
        assert_eq!(
            odd_side.validate().err(),
            Some(SpecError::UnknownSide {
                index: 0,
                side: "encode".into()
            })
        );
        let mut no_plan = PolicySpec::preset("random").expect("preset");
        no_plan.bindings[0].plan = Some("epd".into());
        assert_eq!(
            no_plan.validate().err(),
            Some(SpecError::UnknownPlan {
                index: 0,
                name: "epd".into()
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
        let requests_form = r#"{ "apiVersion": "smg.policy/v1",
          "profiles": { "p2": { "score": ["least-requests"], "pick": "power-of-two" } } }"#;
        let spec: PolicySpec = serde_json::from_str(requests_form).expect("parses");
        let engine = spec.validate().expect("the requests form validates");
        assert!(!engine.needs().loads);
    }

    #[test]
    fn a_spec_without_bindings_binds_its_first_profile_to_everything() {
        let text = r#"{ "apiVersion": "smg.policy/v1",
          "profiles": { "only": { "filter": ["healthy"], "pick": "passthrough" } } }"#;
        let spec: PolicySpec = serde_json::from_str(text).expect("parses");
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

    fn spec_with_score(score: &str) -> PolicySpec {
        serde_json::from_str(&format!(
            r#"{{"apiVersion":"smg.policy/v1","profiles":{{"p":{{"filter":["healthy"],"score":[{score}],"pick":"best"}}}}}}"#
        ))
        .expect("a well-formed document")
    }

    fn spec_with_binding(binding: &str) -> PolicySpec {
        serde_json::from_str(&format!(
            r#"{{"apiVersion":"smg.policy/v1","profiles":{{"p":{{"pick":"best"}}}},"bindings":[{binding}]}}"#
        ))
        .expect("a well-formed document")
    }

    #[test]
    fn a_weight_or_parameter_out_of_range_is_refused_by_name() {
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
            match spec_with_score(score).validate() {
                Err(SpecError::BadParameter {
                    profile,
                    scorer,
                    name: got,
                    ..
                }) => {
                    assert_eq!(
                        (profile.as_str(), scorer.as_str(), got),
                        ("p", "least-load", name),
                        "{score}"
                    );
                }
                other => panic!("{score} validated as {other:?}"),
            }
        }
    }

    #[test]
    fn a_misspelt_scorer_with_parameters_is_refused_as_unknown() {
        match spec_with_score(r#"{"least_load":{"kv_pressure_weight":5}}"#).validate() {
            Err(SpecError::UnknownScorer { name, .. }) => assert_eq!(name, "least_load"),
            other => panic!("validated as {other:?}"),
        }
    }

    #[test]
    fn least_load_parameters_on_another_scorer_are_refused() {
        match spec_with_score(r#"{"least-requests":{"kv_pressure_weight":5}}"#).validate() {
            Err(SpecError::ParameterOutOfPlace { scorer, name, .. }) => {
                assert_eq!(
                    (scorer.as_str(), name),
                    ("least-requests", "kv_pressure_weight")
                );
            }
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
            Some(SpecError::PairPhases {
                index: 0,
                side: "decode"
            })
        );
        let both = spec_with_binding(
            r#"{"match":{"phase":["prefill","decode"]},"plan":"pd-pair","profiles":{"prefill":"p","decode":"p"}}"#,
        );
        assert!(both.validate().is_ok());
    }
}
