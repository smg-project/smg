//! Phase 1b to phase 3 of the start-up, on the parsed arguments: the secrets'
//! sources resolved into the flag fields, the start-up announced, and the
//! server configuration assembled from the flags (`CliArgs::configure`).
//! The argument structs stay with `CliArgs` at the crate root, where clap
//! builds them; this module reads them.

use std::collections::HashMap;

use rand::{distr::Alphanumeric, RngExt};
use smg::{
    config::{
        bind_socket_addr, validate_mesh_server_name, CacheIndexKind, CircuitBreakerConfig,
        ConfigError, ConfigResult, ConsulDiscoveryConfig, DiscoveryConfig, FileDiscoveryConfig,
        HealthCheckConfig, HistoryBackend, KubernetesDiscoveryConfig, ManualAssignmentMode,
        MemoryConfig, MetricsConfig, OracleConfig, ParserChoice, PdPairingMode, PdReadinessConfig,
        PolicyConfig, PostgresConfig, ReadinessRoleConfig, RedisConfig, RetryConfig, RouterConfig,
        RoutingKeyOverrideConfig, RoutingMode, SchemaConfig, SlurmDiscoveryConfig,
        TokenizerCacheConfig, TraceConfig, DEFAULT_DISCOVERY_CHECK_INTERVAL_SECS,
    },
    mesh::discovery::MeshDiscoveryConfig,
    observability::metrics::PrometheusConfig,
    policies::cost::DEFAULT_POLICY as DEFAULT_SELECTION_POLICY,
    server::{posture::SecretSources, ServerConfig},
    service_discovery::RuntimeDiscoveryConfig,
    version,
    worker::{ConnectionMode, RuntimeType},
};
use smg_auth::{ApiKeyEntry, ControlPlaneAuthConfig, JwtConfig, Role};
use smg_mesh::{MTLSConfig, MeshServerConfig};

use super::{
    secrets::{
        EnvLookup, API_KEY_SETTING, POSTGRES_DB_URL_SETTING, REDIS_URL_SETTING,
        TENANT_API_KEYS_SETTING,
    },
    value::{parse_control_plane_api_key, parse_role_mapping, parse_tenant_api_key},
};
use crate::{Backend, CliArgs, DiscoveryProvider};

enum OracleConnectSource {
    Dsn { descriptor: String },
    Wallet { path: String, alias: String },
}

impl CliArgs {
    /// Phase 1b, resolve: the secrets the environment or a file may give in
    /// the command line's place, read into the flag fields so the phases
    /// after this one read flags alone; where each came from is kept for
    /// the start-up record. One rule for every secret: the command line
    /// beats the environment, and a value from either together with a file
    /// is refused rather than ranked. A store URL is read for the selected
    /// history backend alone, as its flag is. The environment lookup is a
    /// parameter so a test need not touch the process environment.
    pub(crate) fn resolve_secret_sources(&mut self, env: EnvLookup<'_>) -> ConfigResult<()> {
        let mut sources = SecretSources::default();
        if let Some((value, source)) = API_KEY_SETTING.resolve_value(
            self.integrations.api_key.as_deref(),
            self.integrations.api_key_file.as_deref(),
            env,
        )? {
            self.integrations.api_key = Some(value);
            sources.api_key = Some(source);
        }
        if let Some((entries, source)) = TENANT_API_KEYS_SETTING.resolve_entries(
            &self.integrations.tenant_api_keys,
            self.integrations.tenant_api_keys_file.as_deref(),
            env,
        )? {
            sources.tenant_api_keys = Some((source, entries.len()));
            self.integrations.tenant_api_keys = entries;
        }
        if self.model.history_backend == "postgres" {
            if let Some((value, source)) = POSTGRES_DB_URL_SETTING.resolve_value(
                self.integrations.postgres_db_url.as_deref(),
                self.integrations.postgres_db_url_file.as_deref(),
                env,
            )? {
                self.integrations.postgres_db_url = Some(value);
                sources.postgres_db_url = Some(source);
            }
        }
        if self.model.history_backend == "redis" {
            if let Some((value, source)) = REDIS_URL_SETTING.resolve_value(
                self.integrations.redis_url.as_deref(),
                self.integrations.redis_url_file.as_deref(),
                env,
            )? {
                self.integrations.redis_url = Some(value);
                sources.redis_url = Some(source);
            }
        }
        self.integrations.secret_sources = sources;
        Ok(())
    }
}

impl CliArgs {
    /// Preserve legacy readiness unless a role threshold was explicitly configured.
    fn pd_readiness_config(&self) -> Option<PdReadinessConfig> {
        let configured = self.routing.readiness_prefill_expected_workers.is_some()
            || self.routing.readiness_prefill_min_healthy_workers.is_some()
            || self.routing.readiness_prefill_min_healthy_percent.is_some()
            || self.routing.readiness_decode_min_healthy_workers.is_some()
            || self.routing.readiness_decode_expected_workers.is_some()
            || self.routing.readiness_decode_min_healthy_percent.is_some();
        configured.then(|| PdReadinessConfig {
            prefill: ReadinessRoleConfig {
                expected_workers: self.routing.readiness_prefill_expected_workers,
                min_healthy_workers: self
                    .routing
                    .readiness_prefill_min_healthy_workers
                    .unwrap_or(1),
                min_healthy_percent: self.routing.readiness_prefill_min_healthy_percent,
            },
            decode: ReadinessRoleConfig {
                expected_workers: self.routing.readiness_decode_expected_workers,
                min_healthy_workers: self
                    .routing
                    .readiness_decode_min_healthy_workers
                    .unwrap_or(1),
                min_healthy_percent: self.routing.readiness_decode_min_healthy_percent,
            },
        })
    }

    /// Build control plane authentication configuration from CLI args. A
    /// configuration that asks for authentication but cannot be honoured
    /// (a malformed key entry, an issuer without an audience or the reverse)
    /// is an error: silently dropping it would start the control plane open.
    pub(crate) fn build_control_plane_auth_config(&self) -> ConfigResult<ControlPlaneAuthConfig> {
        // Build JWT config if issuer and audience are provided
        let jwt = match (
            &self.integrations.jwt_issuer,
            &self.integrations.jwt_audience,
        ) {
            (Some(issuer), Some(audience)) => {
                // One malformed mapping refuses the start: dropped, it could widen access
                let role_mapping = self
                    .integrations
                    .jwt_role_mapping
                    .iter()
                    .map(|m| parse_role_mapping(m))
                    .collect::<ConfigResult<HashMap<String, Role>>>()?;

                let mut jwt_config = JwtConfig::new(issuer.clone(), audience.clone());
                jwt_config
                    .role_claim
                    .clone_from(&self.integrations.jwt_role_claim);
                jwt_config.role_mapping = role_mapping;
                if let Some(jwks_uri) = &self.integrations.jwt_jwks_uri {
                    jwt_config.jwks_uri = Some(jwks_uri.clone());
                }
                Some(jwt_config)
            }
            (Some(_), None) => {
                return Err(ConfigError::IncompatibleConfig {
                    reason: "--jwt-issuer requires --jwt-audience (control-plane JWT authentication would otherwise be silently disabled)".to_string(),
                });
            }
            (None, Some(_)) => {
                return Err(ConfigError::IncompatibleConfig {
                    reason: "--jwt-audience requires --jwt-issuer (control-plane JWT authentication would otherwise be silently disabled)".to_string(),
                });
            }
            (None, None) => None,
        };

        // Build API keys from CLI args; one malformed entry refuses the start
        let api_keys = self
            .integrations
            .control_plane_api_keys
            .iter()
            .map(|k| parse_control_plane_api_key(k))
            .collect::<ConfigResult<Vec<ApiKeyEntry>>>()?;

        Ok(ControlPlaneAuthConfig {
            jwt,
            api_keys,
            audit_enabled: !self.integrations.disable_audit_logging,
        })
    }

    fn determine_connection_mode(worker_urls: &[String]) -> ConnectionMode {
        // First worker URL that declares ipc:// or grpc:// wins; http:// and bare
        // host:port fall through to the HTTP default. See ConnectionMode::from_url.
        worker_urls
            .iter()
            .find_map(|url| match ConnectionMode::from_url(url) {
                mode @ (Some(ConnectionMode::Zmq) | Some(ConnectionMode::Grpc)) => mode,
                _ => None,
            })
            .unwrap_or(ConnectionMode::Http)
    }

    /// The worker discovery provider selected by either spelling:
    /// `--service-discovery` is `--discovery-provider kubernetes`.
    fn selected_discovery_provider(&self) -> Option<DiscoveryProvider> {
        if self.discovery.service_discovery {
            Some(DiscoveryProvider::Kubernetes)
        } else {
            self.discovery.discovery_provider
        }
    }

    /// Selecting a discovery provider, by either spelling, turns IGW mode on.
    /// Returns whether this call turned it on.
    pub(crate) fn enable_igw_for_discovery(&mut self) -> bool {
        let enable = self.selected_discovery_provider().is_some() && !self.routing.enable_igw;
        self.routing.enable_igw |= enable;
        enable
    }

    fn parse_selector(selector_list: &[String]) -> HashMap<String, String> {
        let mut map = HashMap::new();
        for item in selector_list {
            if let Some(eq_pos) = item.find('=') {
                let key = item[..eq_pos].to_string();
                let value = item[eq_pos + 1..].to_string();
                map.insert(key, value);
            }
        }
        map
    }

    /// The metrics bind host: `--prometheus-host`, or the unspecified address
    /// of `--host`'s family when it is not given.
    pub(crate) fn metrics_host(&self) -> String {
        self.serving
            .prometheus_host
            .clone()
            .unwrap_or_else(|| MetricsConfig::default_host_for(&self.routing.host))
    }

    fn parse_mesh_socket_addr(
        host: &str,
        port: u16,
        field: &str,
    ) -> ConfigResult<std::net::SocketAddr> {
        bind_socket_addr(host, port).map_err(|e| ConfigError::InvalidValue {
            field: field.to_string(),
            value: host.to_string(),
            reason: format!("invalid mesh socket address: {e}"),
        })
    }

    /// Build mesh settings from the integration flags, validating addresses and TLS pairs.
    pub(crate) fn build_mesh_server_config(&self) -> ConfigResult<Option<MeshServerConfig>> {
        if !self.integrations.enable_mesh {
            return Ok(None);
        }

        let self_name = if let Some(name) = &self.integrations.mesh_server_name {
            validate_mesh_server_name(name)?;
            name.to_string()
        } else {
            let mut rng = rand::rng();
            let random_string: String = (0..4).map(|_| rng.sample(Alphanumeric) as char).collect();
            format!("Mesh_{random_string}")
        };

        let peer = self
            .integrations
            .mesh_peer_urls
            .first()
            .map(|url| {
                url.parse::<std::net::SocketAddr>()
                    .map_err(|e| ConfigError::InvalidValue {
                        field: "mesh_peer_urls".to_string(),
                        value: url.clone(),
                        reason: format!("invalid socket address: {e}"),
                    })
            })
            .transpose()?;

        // Port 0 parses fine as a socket address, but it is meaningless as an
        // advertised one: peers gossip this port and cannot dial an ephemeral
        // bind. It would also reach router discovery as the fallback mesh port
        // for Pods whose annotation is missing or invalid, publishing `IP:0`
        // peers into the cluster state.
        if self.integrations.mesh_port == 0 {
            return Err(ConfigError::InvalidValue {
                field: "mesh_port".to_string(),
                value: self.integrations.mesh_port.to_string(),
                reason: "mesh port cannot be 0; peers dial the advertised port, \
                         so it must be a fixed, routable one"
                    .to_string(),
            });
        }

        // The mesh port against the other listeners' is checked in
        // `server::startup`, for every way the server configuration is built.
        let bind_addr = Self::parse_mesh_socket_addr(
            &self.integrations.mesh_host,
            self.integrations.mesh_port,
            "mesh_host",
        )?;
        let (advertise_host, advertise_field) =
            if let Some(host) = self.integrations.mesh_advertise_host.as_deref() {
                (host, "mesh_advertise_host")
            } else {
                (self.integrations.mesh_host.as_str(), "mesh_host")
            };
        let advertise_addr = Self::parse_mesh_socket_addr(
            advertise_host,
            self.integrations.mesh_port,
            advertise_field,
        )?;

        if advertise_addr.ip().is_unspecified() {
            return Err(ConfigError::InvalidValue {
                field: advertise_field.to_string(),
                value: advertise_host.to_string(),
                reason:
                    "mesh advertise address cannot be unspecified; set --mesh-advertise-host to a routable node IP".to_string(),
            });
        }

        Ok(Some(MeshServerConfig {
            self_name,
            bind_addr,
            advertise_addr,
            init_peer: peer,
            mtls_config: self.mesh_mtls_config()?,
        }))
    }

    /// The mesh mTLS configuration from `--mesh-tls-ca-cert`, `--mesh-tls-cert`
    /// and `--mesh-tls-key` (all three or none, which the parser enforces);
    /// each file has to be readable at start-up, not at the first dial.
    fn mesh_mtls_config(&self) -> ConfigResult<Option<MTLSConfig>> {
        let (Some(ca), Some(cert), Some(key)) = (
            &self.integrations.mesh_tls_ca_cert,
            &self.integrations.mesh_tls_cert,
            &self.integrations.mesh_tls_key,
        ) else {
            return Ok(None);
        };
        for (field, path) in [
            ("mesh_tls_ca_cert", ca),
            ("mesh_tls_cert", cert),
            ("mesh_tls_key", key),
        ] {
            std::fs::metadata(path).map_err(|e| ConfigError::InvalidValue {
                field: field.to_string(),
                value: path.clone(),
                reason: format!("cannot read the file: {e}"),
            })?;
        }
        Ok(Some(MTLSConfig {
            ca_cert_path: ca.into(),
            server_cert_path: cert.into(),
            server_key_path: key.into(),
            ..MTLSConfig::default()
        }))
    }

    /// Build a routing policy with the same per-policy defaults as the CLI flags.
    fn parse_policy(&self, policy_str: &str) -> PolicyConfig {
        match policy_str {
            "random" => PolicyConfig::Random,
            "round_robin" => PolicyConfig::RoundRobin,
            "passthrough" => PolicyConfig::Passthrough,
            "cache_aware" => PolicyConfig::CacheAware {
                cache_threshold: self.routing.cache_threshold,
                balance_abs_threshold: self.routing.balance_abs_threshold,
                balance_rel_threshold: self.routing.balance_rel_threshold,
                eviction_interval_secs: self.routing.eviction_interval,
                max_tree_size: self.routing.max_tree_size,
                block_size: self.routing.block_size,
                balance_token_usage_threshold: self.routing.balance_token_usage_threshold,
                overload_token_usage_threshold: self.routing.overload_token_usage_threshold,
                overlap_decay: self.routing.overlap_decay,
                selection_temperature: self.routing.selection_temperature,
                cache_index: Self::parse_cache_index(&self.routing.cache_index),
                cache_ttl_secs: self.routing.cache_ttl_secs,
                cache_boundaries: self.routing.cache_boundaries.clone(),
                selection_policy: (self.routing.selection_policy != DEFAULT_SELECTION_POLICY)
                    .then(|| self.routing.selection_policy.clone()),
                selection_accounting_ttl_ms: self.routing.selection_accounting_ttl_ms,
                balance_prefix_margin_tokens: self.routing.balance_prefix_margin_tokens,
                balance_uncached_credit: self.routing.balance_uncached_credit,
            },
            "power_of_two" => PolicyConfig::PowerOfTwo {
                load_metric: self.routing.power_of_two_load_metric,
            },
            "least_load" => PolicyConfig::LeastLoad {
                kv_pressure_weight: self.routing.least_load_kv_pressure_weight,
                mean_prefill_tokens: self.routing.least_load_mean_prefill_tokens,
                default_throughput: self.routing.least_load_default_throughput,
                max_waiting_requests: self.routing.least_load_max_waiting_requests,
            },
            "bucket" => PolicyConfig::Bucket {
                balance_abs_threshold: self.routing.balance_abs_threshold,
                balance_rel_threshold: self.routing.balance_rel_threshold,
                bucket_adjust_interval_secs: self.routing.bucket_adjust_interval_secs,
            },
            "prefix_hash" => PolicyConfig::PrefixHash {
                prefix_token_count: self.routing.prefix_token_count,
                load_factor: self.routing.prefix_hash_load_factor,
                balance_abs_threshold: self.routing.prefix_hash_balance_abs_threshold,
                cache_boundaries: self.routing.cache_boundaries.clone(),
            },
            "consistent_hashing" => PolicyConfig::ConsistentHashing,
            "manual" => PolicyConfig::Manual {
                eviction_interval_secs: self.routing.eviction_interval,
                max_idle_secs: self.routing.max_idle_secs,
                assignment_mode: Self::parse_assignment_mode(
                    self.routing.assignment_mode.as_deref(),
                    ManualAssignmentMode::Random,
                ),
            },
            _ => PolicyConfig::RoundRobin,
        }
    }

    #[expect(
        clippy::panic,
        reason = "unreachable: clap value_parser restricts valid cache index kinds"
    )]
    fn parse_cache_index(kind: &str) -> CacheIndexKind {
        match kind {
            "tree" => CacheIndexKind::Tree,
            "hash" => CacheIndexKind::Hash,
            other => panic!("Unknown cache index: {other}"),
        }
    }

    #[expect(
        clippy::panic,
        reason = "unreachable: clap value_parser restricts valid assignment modes"
    )]
    fn parse_assignment_mode(
        mode: Option<&str>,
        default: ManualAssignmentMode,
    ) -> ManualAssignmentMode {
        let Some(mode) = mode else { return default };
        match mode {
            "random" => ManualAssignmentMode::Random,
            "min_load" => ManualAssignmentMode::MinLoad,
            "min_group" => ManualAssignmentMode::MinGroup,
            "delegate" => ManualAssignmentMode::Delegate,
            other => panic!("Unknown assignment mode: {other}"),
        }
    }

    /// Load the optional storage schema, reporting file and YAML errors as configuration errors.
    fn load_schema_config(&self) -> ConfigResult<Option<SchemaConfig>> {
        match &self.model.schema_config {
            Some(path) => {
                let content =
                    std::fs::read_to_string(path).map_err(|e| ConfigError::ValidationFailed {
                        reason: format!("Failed to read schema config file '{path}': {e}"),
                    })?;
                let schema: SchemaConfig =
                    serde_yaml::from_str(&content).map_err(|e| ConfigError::ValidationFailed {
                        reason: format!("Failed to parse schema config file '{path}': {e}"),
                    })?;
                Ok(Some(schema))
            }
            None => Ok(None),
        }
    }

    /// Prefer an explicit Oracle DSN; otherwise require both wallet path and TNS alias.
    fn resolve_oracle_connect_details(&self) -> ConfigResult<OracleConnectSource> {
        if let Some(dsn) = self.integrations.oracle_dsn.clone() {
            return Ok(OracleConnectSource::Dsn { descriptor: dsn });
        }

        let wallet_path = self
            .integrations
            .oracle_wallet_path
            .clone()
            .ok_or_else(|| ConfigError::MissingRequired {
                field: "oracle_wallet_path or ATP_WALLET_PATH".to_string(),
            })?;

        let tns_alias = self.integrations.oracle_tns_alias.clone().ok_or_else(|| {
            ConfigError::MissingRequired {
                field: "oracle_tns_alias or ATP_TNS_ALIAS".to_string(),
            }
        })?;

        Ok(OracleConnectSource::Wallet {
            path: wallet_path,
            alias: tns_alias,
        })
    }

    /// Build Oracle storage settings after validating authentication and pool bounds.
    fn build_oracle_config(&self, schema: Option<SchemaConfig>) -> ConfigResult<OracleConfig> {
        let (wallet_path, connect_descriptor) = match self.resolve_oracle_connect_details()? {
            OracleConnectSource::Dsn { descriptor } => (None, descriptor),
            OracleConnectSource::Wallet { path, alias } => (Some(path), alias),
        };
        let (username, password) = if self.integrations.oracle_external_auth {
            (
                self.integrations.oracle_user.clone().unwrap_or_default(),
                self.integrations
                    .oracle_password
                    .clone()
                    .unwrap_or_default(),
            )
        } else {
            (
                self.integrations.oracle_user.clone().ok_or_else(|| {
                    ConfigError::MissingRequired {
                        field: "oracle_user or ATP_USER".to_string(),
                    }
                })?,
                self.integrations.oracle_password.clone().ok_or_else(|| {
                    ConfigError::MissingRequired {
                        field: "oracle_password or ATP_PASSWORD".to_string(),
                    }
                })?,
            )
        };

        let pool_min = self
            .integrations
            .oracle_pool_min
            .unwrap_or_else(OracleConfig::default_pool_min);
        let pool_max = self
            .integrations
            .oracle_pool_max
            .unwrap_or_else(OracleConfig::default_pool_max);

        if pool_min == 0 {
            return Err(ConfigError::InvalidValue {
                field: "oracle_pool_min".to_string(),
                value: pool_min.to_string(),
                reason: "pool minimum must be at least 1".to_string(),
            });
        }

        if pool_max < pool_min {
            return Err(ConfigError::InvalidValue {
                field: "oracle_pool_max".to_string(),
                value: pool_max.to_string(),
                reason: "pool maximum must be greater than or equal to minimum".to_string(),
            });
        }

        let pool_timeout_secs = self
            .integrations
            .oracle_pool_timeout_secs
            .unwrap_or_else(OracleConfig::default_pool_timeout_secs);

        Ok(OracleConfig {
            wallet_path,
            connect_descriptor,
            external_auth: self.integrations.oracle_external_auth,
            username,
            password,
            pool_min,
            pool_max,
            pool_timeout_secs,
            schema,
        })
    }

    /// Build PostgreSQL storage settings, requiring a URL and valid pool bounds.
    pub(crate) fn build_postgres_config(
        &self,
        schema: Option<SchemaConfig>,
    ) -> ConfigResult<PostgresConfig> {
        let db_url = self
            .integrations
            .postgres_db_url
            .clone()
            .unwrap_or_default();
        let pool_max = self
            .integrations
            .postgres_pool_max_size
            .unwrap_or_else(PostgresConfig::default_pool_max);
        let pcf = PostgresConfig {
            db_url,
            pool_max,
            op_timeout_secs: self
                .integrations
                .postgres_op_timeout_secs
                .unwrap_or_else(PostgresConfig::default_op_timeout_secs),
            schema,
        };
        pcf.validate().map_err(|e| ConfigError::ValidationFailed {
            reason: e.to_string(),
        })?;
        Ok(pcf)
    }

    /// Build Redis storage settings, requiring a URL and a nonzero pool size.
    pub(crate) fn build_redis_config(
        &self,
        schema: Option<SchemaConfig>,
    ) -> ConfigResult<RedisConfig> {
        let url = self.integrations.redis_url.clone().unwrap_or_default();
        let pool_max = self.integrations.redis_pool_max_size.unwrap_or(16);

        let retention_days = match self.integrations.redis_retention_days {
            Some(-1) => None, // Persistent
            // A retention of 0 days would expire every record as it is
            // written; -1 is the one spelling of persistent records.
            Some(d) if d <= 0 => {
                return Err(ConfigError::InvalidValue {
                    field: "redis_retention_days".to_string(),
                    value: d.to_string(),
                    reason: "Must be at least 1 (-1 keeps records persistent)".to_string(),
                });
            }
            Some(d) => Some(d as u64),
            None => Some(30), // Default 30 days
        };

        let rcf = RedisConfig {
            url,
            pool_max,
            op_timeout_secs: self
                .integrations
                .redis_op_timeout_secs
                .unwrap_or_else(RedisConfig::default_op_timeout_secs),
            retention_days,
            schema,
        };
        rcf.validate().map_err(|e| ConfigError::ValidationFailed {
            reason: e.to_string(),
        })?;
        Ok(rcf)
    }

    /// The flags whose value has a required shape and whose raw form never
    /// reaches `RouterConfig`, refused by name when the shape is wrong: a
    /// wrong shape used to be dropped or passed on in silence. The shapes of
    /// the fields `RouterConfig` carries (the CORS origins, the health-check
    /// endpoint, the request-id header names, the storage-context header
    /// names, an empty API key) are checked by `ConfigValidator`, on every
    /// construction path.
    fn validate_flag_formats(&self) -> ConfigResult<()> {
        let invalid = |field: &str, value: &str, reason: &str| ConfigError::InvalidValue {
            field: field.to_string(),
            value: value.to_string(),
            reason: reason.to_string(),
        };
        // `cors_allowed_origins`, the health-check endpoint, the request-id
        // header names and an empty API key are checked by the shared
        // `ConfigValidator`, which every construction path runs (this CLI
        // through `builder.build()`).
        let buckets = &self.serving.prometheus_duration_buckets;
        if let Some(bad) = buckets.iter().find(|b| !b.is_finite() || **b <= 0.0) {
            return Err(invalid(
                "prometheus_duration_buckets",
                &bad.to_string(),
                "every bucket bound must be a finite number above 0",
            ));
        }
        if let Some(pair) = buckets.windows(2).find(|w| w[1] <= w[0]) {
            return Err(invalid(
                "prometheus_duration_buckets",
                &format!("{} {}", pair[0], pair[1]),
                "bucket bounds must ascend strictly",
            ));
        }
        for entry in &self.serving.storage_context_headers {
            if !entry.contains('=') {
                return Err(invalid(
                    "storage_context_headers",
                    entry,
                    "expected 'header-name=context-key'",
                ));
            }
        }
        for mapping in &self.integrations.jwt_role_mapping {
            match mapping.split_once('=') {
                Some((idp_role, gateway_role))
                    if !idp_role.is_empty()
                        && matches!(
                            gateway_role.to_ascii_lowercase().as_str(),
                            "admin" | "user"
                        ) => {}
                _ => {
                    return Err(invalid(
                        "jwt_role_mapping",
                        mapping,
                        "expected 'idp_role=admin' or 'idp_role=user'",
                    ))
                }
            }
        }
        Ok(())
    }

    /// The flags that only act with a companion, named when the companion is
    /// absent: each used to be ignored in silence. One row per warning, in
    /// the order they are printed: when it applies, and its text.
    pub(crate) fn ignored_companions(
        &self,
        prefill_urls: &[(String, Option<u16>)],
        encode_urls: &[(String, Option<u16>)],
    ) -> Vec<String> {
        const PD: &str = "without --pd-disaggregation (or --epd-disaggregation)";
        const EPD: &str = "without --epd-disaggregation";
        let pd = self.workers.pd_disaggregation || self.workers.epd_disaggregation;
        let epd = self.workers.epd_disaggregation;
        // The selectors select Kubernetes pods: another provider ignores them
        // as the absence of any does.
        let provider = self.selected_discovery_provider();
        let kubernetes = provider == Some(DiscoveryProvider::Kubernetes);
        let selectors = match provider {
            None => "without --service-discovery".to_string(),
            Some(provider) => format!(
                "under --discovery-provider {}: it selects Kubernetes pods",
                provider.flag_value()
            ),
        };
        // Two worker sources are both honoured, which is worth a line of its
        // own, naming the source as it was given.
        let source = if self.discovery.service_discovery {
            "--service-discovery".to_string()
        } else {
            provider.map_or_else(String::new, |provider| {
                format!("--discovery-provider {}", provider.flag_value())
            })
        };
        let ignored = |flag: &str, needs: &str| format!("{flag} is ignored {needs}");
        let warnings = [
            (!prefill_urls.is_empty() && !pd, ignored("--prefill", PD)),
            (
                !self.workers.decode.is_empty() && !pd,
                ignored("--decode", PD),
            ),
            (
                self.workers.prefill_policy.is_some() && !pd,
                ignored("--prefill-policy", PD),
            ),
            (
                self.workers.decode_policy.is_some() && !pd,
                ignored("--decode-policy", PD),
            ),
            (!encode_urls.is_empty() && !epd, ignored("--encode", EPD)),
            (
                self.workers.encode_policy.is_some() && !epd,
                ignored("--encode-policy", EPD),
            ),
            (
                !self.discovery.selector.is_empty() && !kubernetes,
                ignored("--selector", selectors.as_str()),
            ),
            (
                !self.discovery.prefill_selector.is_empty() && !kubernetes,
                ignored("--prefill-selector", selectors.as_str()),
            ),
            (
                !self.discovery.decode_selector.is_empty() && !kubernetes,
                ignored("--decode-selector", selectors.as_str()),
            ),
            (
                !self.discovery.encode_selector.is_empty() && !kubernetes,
                ignored("--encode-selector", selectors.as_str()),
            ),
            (
                !self.discovery.router_selector.is_empty() && !kubernetes,
                ignored("--router-selector", selectors.as_str()),
            ),
            (
                provider.is_some() && !self.routing.worker_urls.is_empty(),
                format!(
                    "--worker-urls and {source} are both given: both are worker sources, \
                     the listed workers register first and the discovered ones join them"
                ),
            ),
            (
                self.serving.priority_scheduler_config.is_some()
                    && !self.serving.priority_scheduler_enabled,
                ignored(
                    "--priority-scheduler-config",
                    "without --priority-scheduler-enabled",
                ),
            ),
        ];
        warnings
            .into_iter()
            .filter_map(|(applies, warning)| applies.then_some(warning))
            .collect()
    }

    /// The companion warnings of [`ignored_companions`](Self::ignored_companions), one
    /// line each on stderr: the logger is not up when the flags are read.
    #[expect(clippy::print_stderr, reason = "pre-logger CLI configuration warnings")]
    fn warn_ignored_companions(
        &self,
        prefill_urls: &[(String, Option<u16>)],
        encode_urls: &[(String, Option<u16>)],
    ) {
        for ignored in self.ignored_companions(prefill_urls, encode_urls) {
            eprintln!("WARNING: {ignored}");
        }
    }

    /// The routing mode the flags select. A vendor backend (`--backend
    /// openai|anthropic|gemini`) routes to that vendor whatever else is set;
    /// otherwise `--epd-disaggregation` wins over `--pd-disaggregation`, and
    /// neither is the regular mode over `--worker-urls`. IGW mode does not
    /// change the routing mode: it only affects router initialization.
    fn routing_mode(
        &self,
        prefill_urls: Vec<(String, Option<u16>)>,
        encode_urls: Vec<(String, Option<u16>)>,
    ) -> RoutingMode {
        let policy = |name: Option<&String>| name.map(|p| self.parse_policy(p));
        match (
            self.model.backend,
            self.workers.epd_disaggregation,
            self.workers.pd_disaggregation,
        ) {
            (Some(Backend::Openai), _, _) => RoutingMode::OpenAI {
                worker_urls: self.routing.worker_urls.clone(),
            },
            (Some(Backend::Anthropic), _, _) => RoutingMode::Anthropic {
                worker_urls: self.routing.worker_urls.clone(),
            },
            (Some(Backend::Gemini), _, _) => RoutingMode::Gemini {
                worker_urls: self.routing.worker_urls.clone(),
            },
            (_, true, _) => RoutingMode::EncodePrefillDecode {
                encode_urls,
                prefill_urls,
                decode_urls: self.workers.decode.clone(),
                encode_policy: policy(self.workers.encode_policy.as_ref()),
                prefill_policy: policy(self.workers.prefill_policy.as_ref()),
                decode_policy: policy(self.workers.decode_policy.as_ref()),
            },
            (_, false, true) => RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls: self.workers.decode.clone(),
                prefill_policy: policy(self.workers.prefill_policy.as_ref()),
                decode_policy: policy(self.workers.decode_policy.as_ref()),
            },
            (_, false, false) => RoutingMode::Regular {
                worker_urls: self.routing.worker_urls.clone(),
            },
        }
    }

    /// The discovery configuration of the selected provider, or none. The
    /// Kubernetes flags are ignored under another provider, as they always
    /// were without discovery; the file, Slurm and Consul flags have no other
    /// use, so giving them without their provider is an error, as is any of
    /// those providers without what it cannot run without. The interval applies to whichever
    /// provider is selected.
    fn discovery_config(&self) -> ConfigResult<Option<DiscoveryConfig>> {
        let provider = self.selected_discovery_provider();
        let incompatible = |reason: &str| {
            Err(ConfigError::IncompatibleConfig {
                reason: reason.to_string(),
            })
        };
        if provider != Some(DiscoveryProvider::File) && self.discovery.discovery_file.is_some() {
            return incompatible("--discovery-file applies only to --discovery-provider file");
        }
        if provider.is_none() && self.discovery.discovery_check_interval_secs.is_some() {
            return incompatible("--discovery-check-interval-secs needs a discovery provider");
        }
        if provider != Some(DiscoveryProvider::Slurm) && self.has_slurm_flags() {
            return incompatible("The --slurm-* flags apply only to --discovery-provider slurm");
        }
        if provider != Some(DiscoveryProvider::Consul) && self.has_consul_flags() {
            return incompatible("The --consul-* flags apply only to --discovery-provider consul");
        }
        let interval = self
            .discovery
            .discovery_check_interval_secs
            .unwrap_or(DEFAULT_DISCOVERY_CHECK_INTERVAL_SECS);
        Ok(match provider {
            None => None,
            Some(DiscoveryProvider::Kubernetes) => {
                Some(DiscoveryConfig::Kubernetes(KubernetesDiscoveryConfig {
                    namespace: self.discovery.service_discovery_namespace.clone(),
                    port: self.discovery.service_discovery_port,
                    check_interval_secs: interval,
                    selector: Self::parse_selector(&self.discovery.selector),
                    encode_selector: Self::parse_selector(&self.discovery.encode_selector),
                    prefill_selector: Self::parse_selector(&self.discovery.prefill_selector),
                    decode_selector: Self::parse_selector(&self.discovery.decode_selector),
                    bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
                    worker_ports_annotation: "smg.ai/worker-ports".to_string(),
                    kv_connector_annotation: self.discovery.kv_connector_annotation.clone(),
                    kv_engine_id_annotation: self.discovery.kv_engine_id_annotation.clone(),
                    router_selector: Self::parse_selector(&self.discovery.router_selector),
                    router_mesh_port_annotation: "sglang.ai/mesh-port".to_string(),
                    model_id_source: self.discovery.model_id_from.clone(),
                }))
            }
            Some(DiscoveryProvider::File) => Some(DiscoveryConfig::File(FileDiscoveryConfig {
                path: self.discovery.discovery_file.clone().ok_or_else(|| {
                    ConfigError::MissingRequired {
                        field: "--discovery-file".to_string(),
                    }
                })?,
                check_interval_secs: interval,
            })),
            Some(DiscoveryProvider::Slurm) => Some(DiscoveryConfig::Slurm(SlurmDiscoveryConfig {
                job_names: self.discovery.slurm_job_name.clone(),
                encode_job_names: self.discovery.slurm_encode_job_name.clone(),
                prefill_job_names: self.discovery.slurm_prefill_job_name.clone(),
                decode_job_names: self.discovery.slurm_decode_job_name.clone(),
                port: self
                    .discovery
                    .slurm_port
                    .ok_or_else(|| ConfigError::MissingRequired {
                        field: "--slurm-port".to_string(),
                    })?,
                bootstrap_port: self.discovery.slurm_bootstrap_port,
                user: self.discovery.slurm_user.clone(),
                account: self.discovery.slurm_account.clone(),
                partition: self.discovery.slurm_partition.clone(),
                check_interval_secs: interval,
            })),
            Some(DiscoveryProvider::Consul) => {
                Some(DiscoveryConfig::Consul(ConsulDiscoveryConfig {
                    service: self.discovery.consul_service.clone().ok_or_else(|| {
                        ConfigError::MissingRequired {
                            field: "--consul-service".to_string(),
                        }
                    })?,
                    address: self.discovery.consul_address.clone(),
                    datacenter: self.discovery.consul_datacenter.clone(),
                    check_interval_secs: interval,
                }))
            }
        })
    }

    /// Whether any `--consul-*` flag was given.
    fn has_consul_flags(&self) -> bool {
        self.discovery.consul_service.is_some()
            || self.discovery.consul_address.is_some()
            || self.discovery.consul_datacenter.is_some()
    }

    /// Whether any `--slurm-*` flag was given.
    fn has_slurm_flags(&self) -> bool {
        [
            &self.discovery.slurm_job_name,
            &self.discovery.slurm_encode_job_name,
            &self.discovery.slurm_prefill_job_name,
            &self.discovery.slurm_decode_job_name,
        ]
        .iter()
        .any(|names| !names.is_empty())
            || self.discovery.slurm_port.is_some()
            || self.discovery.slurm_bootstrap_port.is_some()
            || self.discovery.slurm_user.is_some()
            || self.discovery.slurm_account.is_some()
            || self.discovery.slurm_partition.is_some()
    }

    /// Every worker url the mode starts with, in the order the flags list
    /// them (encode, prefill, decode for the disaggregated modes): the first
    /// url that declares a wire protocol decides the connection mode.
    fn startup_worker_urls(mode: &RoutingMode) -> Vec<String> {
        match mode {
            RoutingMode::Regular { worker_urls }
            | RoutingMode::OpenAI { worker_urls }
            | RoutingMode::Anthropic { worker_urls }
            | RoutingMode::Gemini { worker_urls } => worker_urls.clone(),
            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls,
                ..
            } => prefill_urls
                .iter()
                .map(|(url, _)| url.clone())
                .chain(decode_urls.iter().cloned())
                .collect(),
            RoutingMode::EncodePrefillDecode {
                encode_urls,
                prefill_urls,
                decode_urls,
                ..
            } => encode_urls
                .iter()
                .chain(prefill_urls.iter())
                .map(|(url, _)| url.clone())
                .chain(decode_urls.iter().cloned())
                .collect(),
        }
    }

    /// `--backend` normally only steers the routing mode. Over ZMQ it
    /// additionally pins the startup workers' runtime: the shared EngineCore
    /// handshake carries no engine identity, so the wire protocol cannot be
    /// probed and must be declared. HTTP/gRPC keep auto-detection (None).
    fn startup_worker_runtime_type(&self, connection_mode: ConnectionMode) -> Option<RuntimeType> {
        if connection_mode != ConnectionMode::Zmq {
            return None;
        }
        match self.model.backend {
            Some(Backend::Vllm) => Some(RuntimeType::Vllm),
            Some(Backend::Tokenspeed) => Some(RuntimeType::TokenSpeed),
            Some(Backend::Sglang) => Some(RuntimeType::Sglang),
            _ => None,
        }
    }

    /// The `--model-alias` entries as a map. clap validated each entry's
    /// shape; here only the same alias naming two different canonical models
    /// is refused, which would make the routing outcome depend on argument
    /// order.
    fn model_aliases(entries: &[String]) -> ConfigResult<HashMap<String, String>> {
        let mut model_aliases: HashMap<String, String> = HashMap::new();
        for entry in entries {
            let Some((alias, canonical)) = entry.split_once('=') else {
                return Err(ConfigError::InvalidValue {
                    field: "model_alias".to_string(),
                    value: entry.clone(),
                    reason: "Expected: <alias>=<canonical>".to_string(),
                });
            };
            if let Some(previous) = model_aliases.insert(alias.to_string(), canonical.to_string()) {
                if previous != canonical {
                    return Err(ConfigError::InvalidValue {
                        field: "model_alias".to_string(),
                        value: alias.to_string(),
                        reason: format!("Alias maps to both '{previous}' and '{canonical}'"),
                    });
                }
            }
        }
        Ok(model_aliases)
    }

    /// The router configuration of the flags. A refusal is raised in this
    /// order: the flag shapes, the discovery flags, the schema file, the
    /// tenant keys, the history store, the model aliases, and last the
    /// shared validator over the built configuration, which the builder
    /// runs.
    pub(crate) fn to_router_config(
        &self,
        prefill_urls: Vec<(String, Option<u16>)>,
        encode_urls: Vec<(String, Option<u16>)>,
    ) -> ConfigResult<RouterConfig> {
        self.validate_flag_formats()?;
        self.warn_ignored_companions(&prefill_urls, &encode_urls);

        let mode = self.routing_mode(prefill_urls, encode_urls);
        let policy = self.parse_policy(&self.routing.policy);
        let discovery = self.discovery_config()?;
        let metrics = Some(MetricsConfig {
            port: self.serving.prometheus_port,
            host: self.metrics_host(),
        });
        let trace_config = Some(TraceConfig {
            enable_trace: self.integrations.enable_trace,
            otlp_traces_endpoint: self.integrations.otlp_traces_endpoint.clone(),
        });
        let connection_mode = Self::determine_connection_mode(&Self::startup_worker_urls(&mode));
        let startup_worker_runtime_type = self.startup_worker_runtime_type(connection_mode);
        let history_backend = match self.model.history_backend.as_str() {
            "none" => HistoryBackend::None,
            "oracle" => HistoryBackend::Oracle,
            "postgres" => HistoryBackend::Postgres,
            "redis" => HistoryBackend::Redis,
            _ => HistoryBackend::Memory,
        };
        let schema = self.load_schema_config()?;
        let tenant_api_keys = self
            .integrations
            .tenant_api_keys
            .iter()
            .map(|k| parse_tenant_api_key(k))
            .collect::<ConfigResult<Vec<_>>>()?;
        let (oracle, postgres, redis) = match history_backend {
            HistoryBackend::Oracle => (Some(self.build_oracle_config(schema)?), None, None),
            HistoryBackend::Postgres => (None, Some(self.build_postgres_config(schema)?), None),
            HistoryBackend::Redis => (None, None, Some(self.build_redis_config(schema)?)),
            _ => (None, None, None),
        };
        let model_aliases = Self::model_aliases(&self.discovery.model_alias)?;

        let builder = RouterConfig::builder()
            .mode(mode)
            .policy(policy)
            .cache_boundaries(self.routing.cache_boundaries.clone())
            .connection_mode(connection_mode)
            .startup_worker_runtime_type(startup_worker_runtime_type)
            .zmq_engine_count(self.routing.zmq_engine_count)
            .zmq_startup_timeout_secs(self.routing.zmq_startup_timeout_secs)
            .host(&self.routing.host)
            .port(self.routing.port)
            .pd_readiness(self.pd_readiness_config())
            .health_check_port(self.routing.health_check_port)
            .admin_port(self.routing.admin_port)
            .admin_host(self.routing.admin_host.clone())
            .runtime_worker_threads(self.integrations.runtime_worker_threads)
            .max_payload_size(self.serving.max_payload_size)
            .max_buffered_request_bytes(self.serving.max_buffered_request_bytes)
            .stream_body_stall_timeout_secs(self.serving.stream_body_stall_timeout_secs)
            .header_read_timeout_secs(self.serving.header_read_timeout_secs)
            .idle_connection_timeout_secs(self.serving.idle_connection_timeout_secs)
            .max_connections(self.serving.max_connections)
            .request_timeout_secs(self.serving.request_timeout_secs)
            .grpc_request_timeout_secs(self.serving.grpc_request_timeout_secs)
            .sse_keepalive_secs(self.serving.sse_keepalive_secs)
            .upstream_pool_idle_timeout_secs(self.serving.upstream_pool_idle_timeout_secs)
            .worker_startup_timeout_secs(self.workers.worker_startup_timeout_secs)
            .worker_startup_delay_secs(self.workers.worker_startup_delay)
            .worker_startup_check_interval_secs(self.workers.worker_startup_check_interval)
            .job_queue_capacity(self.routing.job_queue_capacity)
            .job_queue_concurrency(self.routing.job_queue_concurrency)
            .load_monitor_interval_secs(self.workers.load_monitor_interval)
            .worker_stall_secs(self.workers.worker_stall_secs)
            .worker_wedge_secs(self.workers.worker_wedge_secs)
            .worker_stale_secs(self.workers.worker_stale_secs)
            .worker_warmup(
                self.routing.worker_warmup_secs,
                self.routing.worker_warmup_share,
                self.routing.worker_warmup_blocks,
                self.routing.worker_warmup_thin_ratio,
                self.routing.worker_warmup_divert_every,
            )
            .pd_admission_wait_secs(self.workers.pd_admission_wait_secs)
            .disable_load_monitoring(self.workers.disable_load_monitoring)
            .worker_overload_protection(
                self.routing.worker_overload_protection
                    && !self.routing.disable_worker_overload_protection,
            )
            .worker_overload_waiting_requests(Some(self.routing.worker_overload_waiting_requests))
            .worker_overload_token_usage(Some(self.routing.worker_overload_token_usage))
            .worker_overload_shed(self.routing.worker_overload_shed)
            .engine_metrics(self.workers.engine_metrics)
            .multimodal_tensor_transport(self.workers.multimodal_tensor_transport)
            .multimodal_shm_min_bytes(self.workers.multimodal_shm_min_bytes)
            .multimodal_max_inflight_bytes(self.workers.multimodal_max_inflight_bytes)
            .mm_per_request_image_limit(self.workers.mm_per_request_image_limit.map(|v| v as usize))
            .mm_processing(self.workers.mm_processing)
            .mm_pixel_cache_mb(self.workers.mm_pixel_cache_mb)
            .mm_pixel_rdma(self.workers.mm_pixel_rdma)
            .mm_allowed_media_domains(self.workers.mm_allowed_media_domains.clone())
            .mm_allow_private_media_hosts(self.workers.mm_allow_private_media_hosts)
            .mm_turbojpeg(self.workers.mm_turbojpeg)
            .mm_video_fetch_timeout_ms(self.workers.mm_video_fetch_timeout_ms)
            .mm_fetch_timeout_ms(self.workers.mm_fetch_timeout_ms)
            .rdma_listen_ip(self.workers.rdma_listen_ip.clone())
            .rdma_slot_ttl_s(self.workers.rdma_slot_ttl_s)
            .log_mm_timing(self.workers.log_mm_timing)
            .max_concurrent_requests(self.serving.max_concurrent_requests)
            .queue_size(self.serving.queue_size)
            .queue_timeout_secs(self.serving.queue_timeout_secs)
            .prefill_max_inflight_requests_per_worker(
                self.workers.prefill_max_inflight_requests_per_worker,
            )
            .prefill_queue_size(self.workers.prefill_queue_size)
            .prefill_queue_timeout_secs(self.workers.prefill_queue_timeout_secs)
            .priority_scheduler_enabled(self.serving.priority_scheduler_enabled)
            .priority_scheduler_default_max_class(
                self.serving.priority_scheduler_default_max_class.clone(),
            )
            .priority_scheduler_config(self.serving.priority_scheduler_config.clone())
            .priority_scheduler_tenant_metric_top_n(
                self.serving.priority_scheduler_tenant_metric_top_n,
            )
            .tenant_rate_limit_enabled(self.serving.tenant_rate_limit_enabled)
            .tenant_rate_limit_config(self.serving.tenant_rate_limit_config.clone())
            .cors_allowed_origins(self.serving.cors_allowed_origins.clone())
            .retry_config(RetryConfig {
                max_retries: self.serving.retry_max_retries,
                initial_backoff_ms: self.serving.retry_initial_backoff_ms,
                max_backoff_ms: self.serving.retry_max_backoff_ms,
                backoff_multiplier: self.serving.retry_backoff_multiplier,
                jitter_factor: self.serving.retry_jitter_factor,
            })
            .circuit_breaker_config(CircuitBreakerConfig {
                failure_threshold: self.serving.cb_failure_threshold,
                success_threshold: self.serving.cb_success_threshold,
                timeout_duration_secs: self.serving.cb_timeout_duration_secs,
                window_duration_secs: self.serving.cb_window_duration_secs,
            })
            .health_check_config(HealthCheckConfig {
                failure_threshold: self.serving.health_failure_threshold,
                success_threshold: self.serving.health_success_threshold,
                timeout_secs: self.serving.health_check_timeout_secs,
                check_interval_secs: self.serving.health_check_interval_secs,
                endpoint: self.serving.health_check_endpoint.clone(),
                disable_health_check: self.serving.disable_health_check,
                // Kept unresolved: the server resolves it against the
                // discovery the configuration selects.
                remove_unhealthy_workers: self.serving.remove_unhealthy_workers,
                drain_settle_secs: self.serving.drain_settle_secs,
            })
            .tokenizer_cache(TokenizerCacheConfig {
                enable_l0: self.model.tokenizer_cache_enable_l0,
                l0_max_entries: self.model.tokenizer_cache_l0_max_entries,
                l0_max_memory: self.model.tokenizer_cache_l0_max_memory,
                enable_l1: self.model.tokenizer_cache_enable_l1,
                l1_max_memory: self.model.tokenizer_cache_l1_max_memory,
            })
            .disable_tokenizer_autoload(self.model.disable_tokenizer_autoload)
            .parser(ParserChoice::parse(&self.model.parser).unwrap_or_default())
            .history_backend(history_backend)
            .memory_config(MemoryConfig {
                max_responses: self.model.history_memory_max_responses,
                max_items: self.model.history_memory_max_items,
                retention_secs: (self.model.history_memory_retention_secs > 0)
                    .then_some(self.model.history_memory_retention_secs),
            })
            .log_level(&self.serving.log_level)
            .access_log_fields(self.serving.access_log_fields.clone())
            .maybe_api_key(self.integrations.api_key.as_ref())
            .tenant_api_keys(tenant_api_keys)
            .maybe_discovery(discovery)
            .maybe_metrics(metrics)
            .maybe_trace(trace_config)
            .maybe_log_dir(self.serving.log_dir.as_ref())
            .maybe_jemalloc_prof_dir(self.serving.jemalloc_prof_dir.as_ref())
            .maybe_request_id_headers(
                (!self.serving.request_id_headers.is_empty())
                    .then(|| self.serving.request_id_headers.clone()),
            )
            .maybe_storage_context_headers(
                (!self.serving.storage_context_headers.is_empty())
                    .then(|| Self::parse_selector(&self.serving.storage_context_headers)),
            )
            .trust_tenant_header(self.serving.trust_tenant_header)
            .tenant_header_name(&self.serving.tenant_header_name)
            .maybe_rate_limit_tokens_per_second(self.serving.rate_limit_tokens_per_second)
            .max_background_responses(self.serving.max_background_responses)
            .maybe_model_path(self.model.model_path.as_ref())
            .maybe_tokenizer_path(self.model.tokenizer_path.as_ref())
            .maybe_chat_template(self.model.chat_template.as_ref())
            .model_aliases(model_aliases)
            .maybe_oracle(oracle)
            .maybe_postgres(postgres)
            .maybe_redis(redis)
            .maybe_reasoning_parser(self.model.reasoning_parser.as_ref())
            .maybe_tool_call_parser(self.model.tool_call_parser.as_ref())
            .maybe_mcp_config_path(self.model.mcp_config_path.as_ref())
            .dp_aware(self.routing.dp_aware)
            .pd_pairing_mode(
                PdPairingMode::parse(&self.routing.pd_pairing_mode).unwrap_or_default(),
            )
            .routing_key_override(RoutingKeyOverrideConfig {
                enabled: self.routing.routing_key_override,
                eviction_interval_secs: self.routing.eviction_interval,
                max_idle_secs: self.routing.max_idle_secs,
                assignment_mode: Self::parse_assignment_mode(
                    self.routing.assignment_mode.as_deref(),
                    ManualAssignmentMode::Delegate,
                ),
                headers: self.routing.routing_key_headers.clone(),
            })
            .retries(!self.serving.disable_retries)
            .upstream_http2(self.routing.upstream_http2)
            .circuit_breaker(!self.serving.disable_circuit_breaker)
            .enable_wasm(self.model.enable_wasm)
            .maybe_storage_hook_wasm_path(self.model.storage_hook_wasm_path.as_deref())
            .igw(self.routing.enable_igw)
            .rl(smg_rl::RlConfig {
                enabled: self.workers.enable_rl,
                control_timeout_secs: self.workers.rl_control_timeout_secs,
                fanout_concurrency: self.workers.rl_fanout_concurrency,
            })
            .dp_minimum_tokens_scheduler(self.routing.dp_minimum_tokens_scheduler)
            .maybe_client_cert_and_key(
                self.integrations.client_cert_path.as_ref(),
                self.integrations.client_key_path.as_ref(),
            )
            .add_ca_certificates(self.integrations.ca_cert_paths.clone())
            .maybe_server_cert_and_key(
                self.integrations.tls_cert_path.as_ref(),
                self.integrations.tls_key_path.as_ref(),
            );

        builder.build()
    }

    /// Combine validated router settings with listener, discovery, and authentication flags.
    pub(crate) fn to_server_config(
        &self,
        router_config: RouterConfig,
    ) -> ConfigResult<ServerConfig> {
        let service_discovery_config = router_config
            .discovery
            .as_ref()
            .map(|discovery| RuntimeDiscoveryConfig::from_config(discovery, &router_config.mode))
            .transpose()?;
        let mesh_discovery_config = router_config
            .discovery
            .as_ref()
            .and_then(MeshDiscoveryConfig::from_discovery);

        let prometheus_config = Some(PrometheusConfig {
            port: self.serving.prometheus_port,
            host: self.metrics_host(),
            duration_buckets: if self.serving.prometheus_duration_buckets.is_empty() {
                None
            } else {
                Some(self.serving.prometheus_duration_buckets.clone())
            },
        });

        // Build control plane auth config
        let control_plane_auth = {
            let config = self.build_control_plane_auth_config()?;
            if config.is_enabled() {
                Some(config)
            } else {
                None
            }
        };

        // ==================== Mesh Server ====================
        let mesh_server_config = self.build_mesh_server_config()?;

        Ok(ServerConfig {
            host: self.routing.host.clone(),
            port: self.routing.port,
            health_check_port: self.routing.health_check_port,
            runtime_worker_threads: self.integrations.runtime_worker_threads,
            router_config,
            max_payload_size: self.serving.max_payload_size,
            log_dir: self.serving.log_dir.clone(),
            log_level: Some(self.serving.log_level.clone()),
            log_json: self.serving.log_json,
            service_discovery_config,
            mesh_discovery_config,
            prometheus_config,
            request_timeout_secs: self.serving.request_timeout_secs,
            request_id_headers: if self.serving.request_id_headers.is_empty() {
                None
            } else {
                Some(self.serving.request_id_headers.clone())
            },
            shutdown_grace_period_secs: self.serving.shutdown_grace_period_secs,
            shutdown_settle_secs: self.serving.shutdown_settle_secs,
            control_plane_auth,
            mesh_server_config,
            webrtc_bind_addr: self.integrations.webrtc_bind_addr,
            webrtc_stun_server: self.integrations.webrtc_stun_server.clone(),
            secret_sources: self.integrations.secret_sources,
        })
    }

    /// The mode label of the start-up banner. It follows the flags, not the
    /// built [`RoutingMode`]: IGW first, then the OpenAI and Anthropic
    /// backends, then the disaggregated modes, then the backend's name
    /// (`--backend gemini` is labelled `Regular (gemini)`, as it always was).
    fn banner_mode(&self) -> String {
        if self.routing.enable_igw {
            "IGW (Inference Gateway)".to_string()
        } else if matches!(self.model.backend, Some(Backend::Openai)) {
            "OpenAI Backend".to_string()
        } else if matches!(self.model.backend, Some(Backend::Anthropic)) {
            "Anthropic Backend".to_string()
        } else if self.workers.epd_disaggregation {
            "EPD Disaggregated".to_string()
        } else if self.workers.pd_disaggregation {
            "PD Disaggregated".to_string()
        } else if let Some(backend) = &self.model.backend {
            format!("Regular ({backend})")
        } else {
            "Regular".to_string()
        }
    }

    /// Phase 2, announce: the banner and the lines under it, printed before
    /// the configuration is built so an operator sees what was asked for
    /// even when it is refused; nothing is changed. The logger is not up
    /// yet: this is plain stdout.
    #[expect(
        clippy::print_stdout,
        reason = "pre-logger startup output: the banner and the flags' summary"
    )]
    pub(crate) fn announce(
        &self,
        igw_enabled_for_discovery: bool,
        prefill_urls: &[(String, Option<u16>)],
        encode_urls: &[(String, Option<u16>)],
    ) {
        if igw_enabled_for_discovery {
            println!("INFO: IGW mode automatically enabled because service discovery is turned on");
        }
        version::print_banner(&self.routing.host, self.routing.port, &self.banner_mode());
        if self.routing.enable_igw {
            return;
        }
        println!("Policy: {}", self.routing.policy);
        if self.workers.pd_disaggregation && !prefill_urls.is_empty() {
            println!("Prefill nodes: {prefill_urls:?}");
            println!("Decode nodes: {:?}", self.workers.decode);
        }
        if self.workers.epd_disaggregation {
            println!("Encode nodes: {encode_urls:?}");
            println!("Prefill nodes: {prefill_urls:?}");
            println!("Decode nodes: {:?}", self.workers.decode);
        }
    }

    /// Phase 3, configure: the router configuration, validated, wrapped in
    /// the server configuration. Every refusal is a [`ConfigError`] naming
    /// the flag; nothing has been bound or spawned when one is raised.
    pub(crate) fn configure(
        &self,
        prefill_urls: Vec<(String, Option<u16>)>,
        encode_urls: Vec<(String, Option<u16>)>,
    ) -> ConfigResult<ServerConfig> {
        let router_config = self.to_router_config(prefill_urls, encode_urls)?;
        router_config.validate()?;
        self.to_server_config(router_config)
    }
}
