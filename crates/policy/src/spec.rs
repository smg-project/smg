//! `PolicySpec`: the one versioned document. Profiles name stages, bindings
//! match requests to profiles, and validation answers what the host must
//! feed before any request exists.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{
    builtins::{
        Best, HasCapacity, HasRoom, Healthy, LeastLoad, LeastRequests, NotOverloaded, NotTried,
        Passthrough, PowerOfTwo, PrefixAffinity, Random, RoundRobin, SessionAffinity,
    },
    framework::{Binding, Engine, Filter, Picker, Profile, Scorer, Weighted},
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
pub struct ProfileSpec {
    #[serde(default)]
    pub filter: Vec<String>,
    #[serde(default)]
    pub score: Vec<ScoreSpec>,
    pub pick: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
pub struct BindingSpec {
    #[serde(rename = "match", default)]
    pub matcher: Match,
    /// The profile of a single decision, and of the prefill side when
    /// `profiles` names no `prefill`.
    #[serde(default)]
    pub profile: Option<String>,
    /// `single` or `pd-pair`.
    #[serde(default)]
    pub plan: Option<String>,
    /// Per-side profiles of the `pd-pair` plan: `prefill`, `decode`.
    #[serde(default)]
    pub profiles: BTreeMap<String, String>,
}

/// The spec document.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    #[error("binding {index}: unknown profile {name:?}")]
    UnknownProfile { index: usize, name: String },
    #[error("binding {index}: unknown plan {name:?}")]
    UnknownPlan { index: usize, name: String },
    #[error("binding {index} names no profile")]
    NoProfile { index: usize },
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

/// The ten preset names of today, as whole profiles. The ones whose stages
/// this build does not carry yet answer `UnknownPreset`.
pub const PRESETS: [&str; 5] = [
    "random",
    "round_robin",
    "passthrough",
    "power_of_two",
    "least_load",
];

impl PolicySpec {
    /// A spec of one preset profile bound to every model and phase.
    pub fn preset(name: &str) -> Result<Self, SpecError> {
        let profile = match name {
            "random" => ProfileSpec {
                filter: vec![
                    "healthy".into(),
                    "not-overloaded".into(),
                    "not-tried".into(),
                ],
                score: vec![],
                pick: "random".into(),
            },
            "round_robin" => ProfileSpec {
                filter: vec![
                    "healthy".into(),
                    "not-overloaded".into(),
                    "not-tried".into(),
                ],
                score: vec![],
                pick: "round-robin".into(),
            },
            "passthrough" => ProfileSpec {
                filter: vec![
                    "healthy".into(),
                    "not-overloaded".into(),
                    "not-tried".into(),
                ],
                score: vec![],
                pick: "passthrough".into(),
            },
            "power_of_two" => ProfileSpec {
                filter: vec![
                    "healthy".into(),
                    "not-overloaded".into(),
                    "not-tried".into(),
                ],
                score: vec![],
                pick: "power-of-two".into(),
            },
            "least_load" => ProfileSpec {
                filter: vec![
                    "healthy".into(),
                    "not-overloaded".into(),
                    "not-tried".into(),
                ],
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
                    let scorer = scorer_by_name(scorer_name, &params).ok_or_else(|| {
                        SpecError::UnknownScorer {
                            profile: name.clone(),
                            name: scorer_name.to_string(),
                        }
                    })?;
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
            let (profile, decode_profile) = match b.plan.as_deref() {
                None | Some("single") => {
                    let name = b
                        .profile
                        .as_deref()
                        .or_else(|| b.profiles.get("single").map(String::as_str))
                        .ok_or(SpecError::NoProfile { index: i })?;
                    (resolve(name)?, None)
                }
                Some("pd-pair") => {
                    let prefill = b
                        .profiles
                        .get("prefill")
                        .map(String::as_str)
                        .or(b.profile.as_deref())
                        .ok_or(SpecError::NoProfile { index: i })?;
                    let decode = b
                        .profiles
                        .get("decode")
                        .map(String::as_str)
                        .unwrap_or(prefill);
                    (resolve(prefill)?, Some(resolve(decode)?))
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
                profile,
                decode_profile,
            });
        }
        if bindings.is_empty() {
            bindings.push(Binding {
                model: "*".into(),
                phases: vec![],
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
            assert_eq!(engine.needs().loads, name == "least_load", "{name}");
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
        let mut wrong = spec;
        wrong.api_version = "smg.policy/v0".into();
        assert!(matches!(wrong.validate(), Err(SpecError::ApiVersion(_))));
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
        assert!(engine
            .binding("m", Phase::Prefill)
            .and_then(|b| b.decode_profile)
            .is_some());
    }
}
