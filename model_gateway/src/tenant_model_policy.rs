//! The per-tenant model allow-list: the model ids a tenant may name on the
//! serving routes, and sees under `/v1/models`.
//!
//! Opt-in through `tenant_model_policy_config`, the path of a YAML document
//! in the shape of the tenant rate limit's: a default policy for every
//! tenant without an entry of its own, and per-tenant entries keyed by the
//! canonical tenant key (`auth:<tenant_id>` of a tenant API key,
//! `header:<id>` of a trusted tenant header). Without the key every tenant
//! may name every model, as before.
//!
//! A model outside the caller's list is refused as a model nobody serves:
//! the 404 `model_not_found` every route answers for an unknown model, in
//! the caller's error envelope. Not a 403: a distinct refusal would tell a
//! tenant which models exist that it may not use, the very thing the list
//! hides, and the public APIs answer the same way (a key without access to
//! a model is told the model does not exist). The gateway log names the
//! tenant and the model, a few lines per tenant per second.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};

use axum::response::Response;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::info;

use crate::{
    config::RouterConfig,
    observability::log_throttle::KeyedLogThrottle,
    routers::error,
    tenant::{authenticated_tenant_key_from_sha256, is_canonical_serving_tenant_key, TenantKey},
    worker::{RuntimeType, WorkerRegistry},
};

/// The on-disk shape: a default policy and the per-tenant entries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TenantModelPolicyYaml {
    /// The list of every tenant without an entry in `tenants`.
    pub default_policy: ModelPolicySpec,
    #[serde(default)]
    pub tenants: Vec<ModelPolicySpec>,
}

/// One tenant's list, or the default: `models` holds the patterns, each an
/// exact model id, a prefix ending in `*` (`org/small-*`), or `*` alone for
/// every model. An empty list admits no model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelPolicySpec {
    /// Canonical tenant key (`auth:team-red`, `header:team-blue`): `None`
    /// on the default policy, required on every `tenants` entry.
    #[serde(default)]
    pub tenant_key: Option<String>,
    pub models: Vec<String>,
}

#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum TenantModelPolicyError {
    #[error("default_policy must not set tenant_key")]
    DefaultPolicyHasTenantKey,
    #[error("tenants[{index}] is missing tenant_key")]
    MissingTenantKey { index: usize },
    #[error("duplicate tenant_key '{tenant_key}'")]
    DuplicateTenantKey { tenant_key: String },
    #[error("tenant_key '{tenant_key}' must not have surrounding whitespace")]
    PaddedTenantKey { tenant_key: String },
    #[error(
        "tenant_key '{tenant_key}' is not a canonical serving-path tenant key (expected auth:<id>, header:<id>, ip:<address>, or anonymous)"
    )]
    NonCanonicalTenantKey { tenant_key: String },
    #[error(
        "tenant_key '{tenant_key}' needs tenant_resolution.trust_tenant_header, which is not set: it can never match a real request"
    )]
    HeaderTenantWithoutTrust { tenant_key: String },
    #[error(
        "tenant_key '{tenant_key}' names no configured tenant API key (no tenant_api_keys entry has tenant_id '{tenant_id}'): it can never match a real request"
    )]
    UnknownAuthTenant {
        tenant_key: String,
        tenant_id: String,
    },
    #[error("policy '{label}': a models entry is empty")]
    EmptyPattern { label: String },
    #[error("policy '{label}': models entry '{pattern}' must not have surrounding whitespace")]
    PaddedPattern { label: String, pattern: String },
    #[error(
        "policy '{label}': models entry '{pattern}' has a '*' that is not at the end (a pattern is an exact model id, a prefix ending in '*', or '*' alone)"
    )]
    WildcardNotAtEnd { label: String, pattern: String },
    #[error("policy '{label}': duplicate models entry '{pattern}'")]
    DuplicatePattern { label: String, pattern: String },
}

/// One tenant's compiled list: a set lookup for the exact ids and a scan of
/// the prefixes per check.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelAllowList {
    every_model: bool,
    exact: HashSet<String>,
    prefixes: Vec<String>,
}

impl ModelAllowList {
    fn compile(label: &str, patterns: &[String]) -> Result<Self, TenantModelPolicyError> {
        let mut list = Self::default();
        let mut seen = HashSet::with_capacity(patterns.len());
        for pattern in patterns {
            let refused = |error: fn(String, String) -> TenantModelPolicyError| {
                error(label.to_owned(), pattern.clone())
            };
            if pattern.is_empty() {
                return Err(TenantModelPolicyError::EmptyPattern {
                    label: label.to_owned(),
                });
            }
            if pattern.trim() != pattern {
                return Err(refused(|label, pattern| {
                    TenantModelPolicyError::PaddedPattern { label, pattern }
                }));
            }
            if !seen.insert(pattern.as_str()) {
                return Err(refused(|label, pattern| {
                    TenantModelPolicyError::DuplicatePattern { label, pattern }
                }));
            }
            if pattern == "*" {
                list.every_model = true;
                continue;
            }
            let (name, is_prefix) = match pattern.strip_suffix('*') {
                Some(prefix) => (prefix, true),
                None => (pattern.as_str(), false),
            };
            if name.contains('*') {
                return Err(refused(|label, pattern| {
                    TenantModelPolicyError::WildcardNotAtEnd { label, pattern }
                }));
            }
            if is_prefix {
                list.prefixes.push(name.to_owned());
            } else {
                list.exact.insert(name.to_owned());
            }
        }
        Ok(list)
    }

    /// Whether the list names `model`, by that id.
    #[must_use]
    pub fn admits(&self, model: &str) -> bool {
        self.every_model
            || self.exact.contains(model)
            || self
                .prefixes
                .iter()
                .any(|prefix| model.starts_with(prefix.as_str()))
    }

    /// Whether the list names a model known by any of `names`: a card's id
    /// and its aliases, a request's id and the canonical id it resolves to.
    pub fn admits_any<'a>(&self, names: impl IntoIterator<Item = &'a str>) -> bool {
        names.into_iter().any(|name| self.admits(name))
    }
}

/// The compiled policy: the default list and the per-tenant lists, built
/// once at start-up and read per request.
#[derive(Debug, Clone)]
pub struct TenantModelPolicy {
    default: ModelAllowList,
    tenants: HashMap<TenantKey, ModelAllowList>,
}

/// A refused request is logged with its tenant and model, a few lines per
/// tenant per second: a client retrying a model it may not use is one
/// condition, not a line per attempt.
static REFUSAL_LOG: KeyedLogThrottle = KeyedLogThrottle::new(2, Duration::from_secs(1), 64);

impl TenantModelPolicy {
    /// Compile `yaml` against `config`. Every tenant key must be canonical
    /// and reachable: a `header:` key needs the trusted tenant header on, an
    /// `auth:<id>` key a tenant API key of that tenant id (or the shared key,
    /// whose tenant id is the digest of the key). A key that can never match
    /// a real request would silently leave its tenant on the default policy,
    /// so it is a configuration error instead.
    pub fn compile(
        yaml: &TenantModelPolicyYaml,
        config: &RouterConfig,
    ) -> Result<Self, TenantModelPolicyError> {
        if yaml.default_policy.tenant_key.is_some() {
            return Err(TenantModelPolicyError::DefaultPolicyHasTenantKey);
        }
        let default = ModelAllowList::compile("default_policy", &yaml.default_policy.models)?;
        let reachable_auth_tenants = reachable_auth_tenant_ids(config);
        let mut tenants = HashMap::with_capacity(yaml.tenants.len());
        for (index, spec) in yaml.tenants.iter().enumerate() {
            let tenant_key = spec
                .tenant_key
                .as_deref()
                .ok_or(TenantModelPolicyError::MissingTenantKey { index })?;
            let refused =
                |error: fn(String) -> TenantModelPolicyError| error(tenant_key.to_owned());
            if tenant_key.trim() != tenant_key {
                return Err(refused(|tenant_key| {
                    TenantModelPolicyError::PaddedTenantKey { tenant_key }
                }));
            }
            if !is_canonical_serving_tenant_key(tenant_key) {
                return Err(refused(|tenant_key| {
                    TenantModelPolicyError::NonCanonicalTenantKey { tenant_key }
                }));
            }
            if tenant_key.starts_with("header:") && !config.tenant_resolution.trust_tenant_header {
                return Err(refused(|tenant_key| {
                    TenantModelPolicyError::HeaderTenantWithoutTrust { tenant_key }
                }));
            }
            if let Some(tenant_id) = tenant_key.strip_prefix("auth:") {
                if !reachable_auth_tenants.contains(tenant_id) {
                    return Err(TenantModelPolicyError::UnknownAuthTenant {
                        tenant_key: tenant_key.to_owned(),
                        tenant_id: tenant_id.to_owned(),
                    });
                }
            }
            let list = ModelAllowList::compile(tenant_key, &spec.models)?;
            if tenants.insert(TenantKey::from(tenant_key), list).is_some() {
                return Err(refused(|tenant_key| {
                    TenantModelPolicyError::DuplicateTenantKey { tenant_key }
                }));
            }
        }
        Ok(Self { default, tenants })
    }

    /// Load and compile the policy `config` names; `Ok(None)` without one.
    pub fn from_config(config: &RouterConfig) -> Result<Option<Arc<Self>>, String> {
        let Some(path) = config.tenant_model_policy_config.as_deref() else {
            return Ok(None);
        };
        let contents = std::fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))?;
        let yaml: TenantModelPolicyYaml =
            serde_yaml::from_str(&contents).map_err(|e| format!("parsing {path}: {e}"))?;
        let policy = Self::compile(&yaml, config).map_err(|e| e.to_string())?;
        Ok(Some(Arc::new(policy)))
    }

    /// The list that applies to `tenant`: its own, or the default.
    #[must_use]
    pub fn allow_list_for(&self, tenant: &TenantKey) -> &ModelAllowList {
        self.tenants.get(tenant).unwrap_or(&self.default)
    }

    /// Whether `tenant` may name `model`. A list names a self-hosted model
    /// under any of the names `registry` knows it by: the id as the request
    /// wrote it, the served id an alias resolves to, or another alias of
    /// that model (an alias a self-hosted worker declares is a second name
    /// of the one model). So a list naming a served id admits its aliases,
    /// and a list naming an alias admits the served id (the one `/v1/models`
    /// lists) and the model's other aliases.
    ///
    /// An external provider's model is named by the id as written, and by
    /// nothing else: its aliases are other models, the dated snapshots
    /// discovery groups under the floating name (see
    /// [`WorkerRegistry::resolve_model_alias`]), so a list naming one of
    /// them names that one alone. A model with an external worker among
    /// those serving it is held to that rule.
    #[must_use]
    pub fn admits(&self, tenant: &TenantKey, model: &str, registry: &WorkerRegistry) -> bool {
        let list = self.allow_list_for(tenant);
        if list.admits(model) {
            return true;
        }
        let canonical = registry.resolve_model_alias(model);
        let canonical = canonical.as_deref().unwrap_or(model);
        let workers = registry.get_by_model(canonical);
        if workers.is_empty()
            || workers
                .iter()
                .any(|worker| worker.metadata().spec.runtime_type == RuntimeType::External)
        {
            return false;
        }
        list.admits(canonical)
            || registry
                .model_aliases(canonical)
                .iter()
                .any(|alias| list.admits(alias))
    }

    /// The refusal of a request of `tenant` naming `model`, or `None` when
    /// the policy admits it: the 404 of a model nobody serves (see the module
    /// docs), logged with the tenant and the model under a per-tenant
    /// allowance.
    #[must_use]
    pub fn refusal(
        &self,
        tenant: &TenantKey,
        model: &str,
        registry: &WorkerRegistry,
    ) -> Option<Response> {
        if self.admits(tenant, model, registry) {
            return None;
        }
        if let Some(admitted) = REFUSAL_LOG.admit(tenant.as_str()) {
            info!(
                tenant = %tenant,
                model,
                suppressed = admitted.suppressed,
                "Model outside the tenant's allow-list, refused as not found"
            );
        }
        Some(error::model_not_found(model))
    }
}

/// The tenant ids an `auth:<id>` key can carry: those of the tenant API keys
/// and, for the shared key, the digest its tenant key is built from.
fn reachable_auth_tenant_ids(config: &RouterConfig) -> HashSet<String> {
    let mut ids: HashSet<String> = config
        .tenant_api_keys
        .iter()
        .map(|entry| entry.tenant_id.clone())
        .collect();
    if let Some(api_key) = &config.api_key {
        let shared =
            authenticated_tenant_key_from_sha256(Sha256::digest(api_key.as_bytes()).into());
        if let Some(id) = shared.as_str().strip_prefix("auth:") {
            ids.insert(id.to_owned());
        }
    }
    ids
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::http::StatusCode;
    use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};

    use super::*;
    use crate::{
        config::{TenantApiKeyEntry, TenantResolutionConfig},
        worker::{BasicWorkerBuilder, RuntimeType, Worker, WorkerType},
    };

    fn config_with_tenants(tenant_ids: &[&str], trust_header: bool) -> RouterConfig {
        RouterConfig {
            tenant_api_keys: tenant_ids
                .iter()
                .map(|id| TenantApiKeyEntry {
                    tenant_id: (*id).to_owned(),
                    key: format!("{id}-secret"),
                })
                .collect(),
            tenant_resolution: TenantResolutionConfig {
                trust_tenant_header: trust_header,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn spec(tenant_key: Option<&str>, models: &[&str]) -> ModelPolicySpec {
        ModelPolicySpec {
            tenant_key: tenant_key.map(str::to_owned),
            models: models.iter().map(|m| (*m).to_owned()).collect(),
        }
    }

    fn yaml(default: &[&str], tenants: Vec<ModelPolicySpec>) -> TenantModelPolicyYaml {
        TenantModelPolicyYaml {
            default_policy: spec(None, default),
            tenants,
        }
    }

    fn compiled(default: &[&str], tenants: Vec<ModelPolicySpec>) -> TenantModelPolicy {
        TenantModelPolicy::compile(
            &yaml(default, tenants),
            &config_with_tenants(&["team-red", "team-blue"], true),
        )
        .expect("a valid policy")
    }

    fn refused(default: &[&str], tenants: Vec<ModelPolicySpec>) -> TenantModelPolicyError {
        TenantModelPolicy::compile(
            &yaml(default, tenants),
            &config_with_tenants(&["team-red", "team-blue"], true),
        )
        .expect_err("an invalid policy")
    }

    /// A registry of one self-hosted worker serving `cards`.
    fn registry_with(cards: Vec<ModelCard>) -> WorkerRegistry {
        registry_of(RuntimeType::Unspecified, cards)
    }

    /// A registry of one worker of `runtime_type` serving `cards`.
    fn registry_of(runtime_type: RuntimeType, cards: Vec<ModelCard>) -> WorkerRegistry {
        let registry = WorkerRegistry::new();
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker.invalid:8000")
                .worker_type(WorkerType::Regular)
                .runtime_type(runtime_type)
                .models(cards)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        registry.register(worker);
        registry
    }

    /// A pattern is an exact id, a prefix ending in `*`, or `*` alone; the
    /// empty list admits nothing.
    #[test]
    fn a_list_admits_its_exact_ids_its_prefixes_or_every_model() {
        let list =
            ModelAllowList::compile("t", &["org/a".to_owned(), "org/small-*".to_owned()]).unwrap();
        assert!(list.admits("org/a"));
        assert!(list.admits("org/small-1"));
        assert!(list.admits("org/small-"));
        assert!(!list.admits("org/a-2"));
        assert!(!list.admits("org/b"));
        assert!(!list.admits("org/smal"));
        assert!(list.admits_any(["org/b", "org/a"]));
        assert!(!list.admits_any(["org/b", "org/c"]));

        let every = ModelAllowList::compile("t", &["*".to_owned()]).unwrap();
        assert!(every.admits("anything"));
        assert!(every.admits(""));

        let none = ModelAllowList::compile("t", &[]).unwrap();
        assert!(!none.admits("org/a"));
        assert!(!none.admits_any(std::iter::empty()));
    }

    #[test]
    fn a_malformed_pattern_is_refused_by_name() {
        let refused = |patterns: &[&str]| {
            ModelAllowList::compile(
                "t",
                &patterns.iter().map(|p| (*p).to_owned()).collect::<Vec<_>>(),
            )
            .expect_err("refused")
        };
        assert_eq!(
            refused(&[""]),
            TenantModelPolicyError::EmptyPattern {
                label: "t".to_owned()
            }
        );
        assert_eq!(
            refused(&[" org/a"]),
            TenantModelPolicyError::PaddedPattern {
                label: "t".to_owned(),
                pattern: " org/a".to_owned()
            }
        );
        assert_eq!(
            refused(&["org/*/a"]),
            TenantModelPolicyError::WildcardNotAtEnd {
                label: "t".to_owned(),
                pattern: "org/*/a".to_owned()
            }
        );
        assert_eq!(
            refused(&["*a*"]),
            TenantModelPolicyError::WildcardNotAtEnd {
                label: "t".to_owned(),
                pattern: "*a*".to_owned()
            }
        );
        assert_eq!(
            refused(&["org/a", "org/a"]),
            TenantModelPolicyError::DuplicatePattern {
                label: "t".to_owned(),
                pattern: "org/a".to_owned()
            }
        );
    }

    /// A tenant with an entry is held to it; every other tenant, the
    /// address-only and anonymous callers included, gets the default.
    #[test]
    fn a_tenant_gets_its_own_list_and_everyone_else_the_default() {
        let policy = compiled(
            &["*"],
            vec![
                spec(Some("auth:team-red"), &["org/a"]),
                spec(Some("header:team-blue"), &[]),
            ],
        );
        let registry = registry_with(vec![ModelCard::new("org/a"), ModelCard::new("org/b")]);
        let red = TenantKey::from("auth:team-red");
        let blue = TenantKey::from("header:team-blue");
        let other = TenantKey::from("auth:team-green");
        let anonymous = TenantKey::from("anonymous");
        let address = TenantKey::from("ip:127.0.0.1");

        assert!(policy.admits(&red, "org/a", &registry));
        assert!(!policy.admits(&red, "org/b", &registry));
        assert!(!policy.admits(&blue, "org/a", &registry));
        for tenant in [&other, &anonymous, &address] {
            assert!(policy.admits(tenant, "org/a", &registry), "{tenant}");
            assert!(policy.admits(tenant, "org/b", &registry), "{tenant}");
        }
        assert!(policy.refusal(&red, "org/a", &registry).is_none());
        let refusal = policy.refusal(&red, "org/b", &registry).expect("refused");
        assert_eq!(refusal.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            refusal
                .headers()
                .get(error::HEADER_X_SMG_ERROR_CODE)
                .unwrap(),
            "model_not_found"
        );
    }

    /// A list names models, under any of their names: a list naming the
    /// served id admits a request for one of its aliases, and a list naming
    /// an alias admits the served id (the one `/v1/models` lists) and the
    /// model's other aliases; a name nobody serves stays refused.
    #[test]
    fn an_alias_is_admitted_through_its_canonical_id() {
        let mut card = ModelCard::new("org/served");
        card.aliases = vec!["short".to_owned(), "other-short".to_owned()];
        let registry = registry_with(vec![card, ModelCard::new("org/other")]);
        let red = TenantKey::from("auth:team-red");

        let by_canonical = compiled(&[], vec![spec(Some("auth:team-red"), &["org/served"])]);
        assert!(by_canonical.admits(&red, "org/served", &registry));
        assert!(by_canonical.admits(&red, "short", &registry));
        assert!(!by_canonical.admits(&red, "org/other", &registry));

        let by_alias = compiled(&[], vec![spec(Some("auth:team-red"), &["short"])]);
        assert!(by_alias.admits(&red, "short", &registry));
        assert!(by_alias.admits(&red, "org/served", &registry));
        assert!(by_alias.admits(&red, "other-short", &registry));
        assert!(!by_alias.admits(&red, "org/other", &registry));
        assert!(!by_alias.admits(&red, "org/nobody", &registry));
    }

    /// An external provider's aliases are not second names of one model:
    /// discovery groups the provider's dated snapshots under the floating
    /// name, and each of them is a model of its own. A list naming one of
    /// them names that one alone: the tenant pinned to a snapshot gets
    /// neither the floating name nor another snapshot, by exact id or by
    /// prefix, and a list naming the floating name admits no snapshot.
    #[test]
    fn an_external_providers_alias_does_not_widen_the_list() {
        let mut floating = ModelCard::new("ext/model");
        floating.aliases = vec![
            "ext/model-2024-05-13".to_owned(),
            "ext/model-2024-08-06".to_owned(),
            "ext/model-2024-11-20".to_owned(),
        ];
        let registry = registry_of(RuntimeType::External, vec![floating]);
        let red = TenantKey::from("auth:team-red");

        let pinned = compiled(
            &[],
            vec![spec(Some("auth:team-red"), &["ext/model-2024-08-06"])],
        );
        assert!(pinned.admits(&red, "ext/model-2024-08-06", &registry));
        assert!(!pinned.admits(&red, "ext/model", &registry));
        assert!(!pinned.admits(&red, "ext/model-2024-05-13", &registry));
        assert!(!pinned.admits(&red, "ext/model-2024-11-20", &registry));

        let by_prefix = compiled(
            &[],
            vec![spec(Some("auth:team-red"), &["ext/model-2024-*"])],
        );
        assert!(by_prefix.admits(&red, "ext/model-2024-05-13", &registry));
        assert!(by_prefix.admits(&red, "ext/model-2024-11-20", &registry));
        assert!(!by_prefix.admits(&red, "ext/model", &registry));

        let floating_only = compiled(&[], vec![spec(Some("auth:team-red"), &["ext/model"])]);
        assert!(floating_only.admits(&red, "ext/model", &registry));
        assert!(!floating_only.admits(&red, "ext/model-2024-08-06", &registry));
        assert!(!floating_only.admits(&red, "ext/nobody", &registry));
    }

    /// A tenant key that can never match a real request is a configuration
    /// error, not a silent fall-through to the default policy.
    #[test]
    fn an_unreachable_or_malformed_tenant_key_is_refused() {
        assert_eq!(
            TenantModelPolicy::compile(
                &TenantModelPolicyYaml {
                    default_policy: spec(Some("auth:team-red"), &["*"]),
                    tenants: vec![],
                },
                &config_with_tenants(&["team-red"], true),
            )
            .unwrap_err(),
            TenantModelPolicyError::DefaultPolicyHasTenantKey
        );
        assert_eq!(
            refused(&["*"], vec![spec(None, &["*"])]),
            TenantModelPolicyError::MissingTenantKey { index: 0 }
        );
        assert_eq!(
            refused(&["*"], vec![spec(Some(" auth:team-red"), &["*"])]),
            TenantModelPolicyError::PaddedTenantKey {
                tenant_key: " auth:team-red".to_owned()
            }
        );
        assert_eq!(
            refused(&["*"], vec![spec(Some("team-red"), &["*"])]),
            TenantModelPolicyError::NonCanonicalTenantKey {
                tenant_key: "team-red".to_owned()
            }
        );
        assert_eq!(
            refused(&["*"], vec![spec(Some("auth:team-rde"), &["*"])]),
            TenantModelPolicyError::UnknownAuthTenant {
                tenant_key: "auth:team-rde".to_owned(),
                tenant_id: "team-rde".to_owned()
            }
        );
        assert_eq!(
            refused(
                &["*"],
                vec![
                    spec(Some("auth:team-red"), &["*"]),
                    spec(Some("auth:team-red"), &["org/a"]),
                ]
            ),
            TenantModelPolicyError::DuplicateTenantKey {
                tenant_key: "auth:team-red".to_owned()
            }
        );
        // A header tenant needs the trusted header switched on.
        let untrusted = config_with_tenants(&["team-red"], false);
        assert_eq!(
            TenantModelPolicy::compile(
                &yaml(&["*"], vec![spec(Some("header:team-blue"), &["*"])]),
                &untrusted,
            )
            .unwrap_err(),
            TenantModelPolicyError::HeaderTenantWithoutTrust {
                tenant_key: "header:team-blue".to_owned()
            }
        );
        // Address and anonymous keys are reachable without any credential.
        TenantModelPolicy::compile(
            &yaml(
                &["*"],
                vec![
                    spec(Some("ip:127.0.0.1"), &["org/a"]),
                    spec(Some("anonymous"), &[]),
                ],
            ),
            &untrusted,
        )
        .expect("reachable keys");
    }

    /// The shared `api_key` authenticates as the digest-derived tenant: its
    /// key is reachable under that id, and under no other.
    #[test]
    fn the_shared_key_is_reachable_under_its_digest_tenant() {
        let mut config = config_with_tenants(&[], false);
        config.api_key = Some("shared-secret".to_owned());
        let shared =
            authenticated_tenant_key_from_sha256(Sha256::digest("shared-secret".as_bytes()).into());
        let policy = TenantModelPolicy::compile(
            &yaml(&[], vec![spec(Some(shared.as_str()), &["org/a"])]),
            &config,
        )
        .expect("the shared key's tenant is reachable");
        let registry = registry_with(vec![ModelCard::new("org/a")]);
        assert!(policy.admits(&shared, "org/a", &registry));
        assert!(!policy.admits(&TenantKey::from("anonymous"), "org/a", &registry));
    }

    /// The YAML is read in the tenant rate limit's shape; an unknown key is
    /// refused rather than ignored (a typo would silently widen access).
    #[test]
    fn the_yaml_is_loaded_from_the_configured_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.yaml");
        std::fs::write(
            &path,
            "default_policy:\n  models: [\"*\"]\ntenants:\n  - tenant_key: auth:team-red\n    models:\n      - org/a\n      - org/small-*\n",
        )
        .unwrap();
        let mut config = config_with_tenants(&["team-red"], false);
        assert!(TenantModelPolicy::from_config(&config).unwrap().is_none());
        config.tenant_model_policy_config = Some(path.to_str().unwrap().to_owned());
        let policy = TenantModelPolicy::from_config(&config)
            .unwrap()
            .expect("a policy");
        let red = TenantKey::from("auth:team-red");
        assert!(policy.allow_list_for(&red).admits("org/small-7"));
        assert!(!policy.allow_list_for(&red).admits("org/b"));
        assert!(policy
            .allow_list_for(&TenantKey::from("anonymous"))
            .admits("org/b"));

        std::fs::write(
            &path,
            "default_policy:\n  models: [\"*\"]\n  routes: [serving]\n",
        )
        .unwrap();
        let reason = TenantModelPolicy::from_config(&config).unwrap_err();
        assert!(
            reason.contains("parsing") && reason.contains("routes"),
            "{reason}"
        );

        config.tenant_model_policy_config =
            Some(dir.path().join("missing.yaml").to_str().unwrap().to_owned());
        let reason = TenantModelPolicy::from_config(&config).unwrap_err();
        assert!(reason.starts_with("reading "), "{reason}");
    }
}
