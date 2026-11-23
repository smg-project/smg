//! The command-line settings and credential for worker discovery.

use std::{
    collections::HashMap,
    ffi::OsString,
    fmt, fs,
    path::{Path, PathBuf},
};

use clap::{
    builder::{StringValueParser, TypedValueParser},
    ArgAction, ValueEnum,
};
use smg_service_discovery::{
    ConsulDiscoveryConfig, DiscoveryConfig, DiscoveryConfigError, FileDiscoveryConfig,
    KubernetesDiscoveryConfig, ModelIdSource, Roles, SlurmDiscoveryConfig,
    DEFAULT_DISCOVERY_CHECK_INTERVAL_SECS,
};

use super::DiscoverySettings;
use crate::service::controller::ConfigError;

const KEY_ENV: &str = "SMG_API_KEY";
const KEY_FILE_ENV: &str = "SMG_API_KEY_FILE";

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Provider {
    Kubernetes,
    File,
    Slurm,
    Consul,
}

/// The worker discovery flags, under the gateway's names.
/// The gateway's peer discovery and routing flags, `--router-selector` and `--model-alias`, stay there.
#[derive(Clone, clap::Args)]
#[group(skip)]
pub(crate) struct DiscoveryFlags {
    /// Enable Kubernetes service discovery (the legacy spelling of
    /// `--discovery-provider kubernetes`)
    #[arg(
        long,
        default_value_t = false,
        help_heading = "Service Discovery (Kubernetes)"
    )]
    service_discovery: bool,

    /// Label selector for Kubernetes service discovery (format: key=value)
    #[arg(long, num_args = 0.., help_heading = "Service Discovery (Kubernetes)")]
    selector: Vec<String>,

    /// Port to use for discovered worker pods without a `smg.ai/worker-ports`
    /// annotation (pods running multiple servers list their ports there)
    #[arg(
        long,
        default_value_t = 80,
        help_heading = "Service Discovery (Kubernetes)"
    )]
    service_discovery_port: u16,

    /// Kubernetes namespace to watch for pods
    #[arg(long, help_heading = "Service Discovery (Kubernetes)")]
    service_discovery_namespace: Option<String>,

    /// Pod annotation containing the worker's KV connector name
    #[arg(
        long,
        default_value = "smg.ai/kv-connector",
        help_heading = "Service Discovery (Kubernetes)"
    )]
    kv_connector_annotation: String,

    /// Pod annotation containing per-worker KV engine IDs
    #[arg(
        long,
        default_value = "smg.ai/kv-engine-id",
        help_heading = "Service Discovery (Kubernetes)"
    )]
    kv_engine_id_annotation: String,

    /// Pod label containing the worker's prefill/decode pairing protocol
    #[arg(
        long,
        default_value = "smg.ai/pairing-protocol",
        help_heading = "Service Discovery (Kubernetes)"
    )]
    pairing_protocol_label: String,

    /// Label selector for encode server pods in EPD mode
    #[arg(long, num_args = 0.., help_heading = "Service Discovery (Kubernetes)")]
    encode_selector: Vec<String>,

    /// Label selector for prefill server pods in PD mode
    #[arg(long, num_args = 0.., help_heading = "Service Discovery (Kubernetes)")]
    prefill_selector: Vec<String>,

    /// Label selector for decode server pods in PD mode
    #[arg(long, num_args = 0.., help_heading = "Service Discovery (Kubernetes)")]
    decode_selector: Vec<String>,

    /// Override each worker's model_id from pod metadata.
    /// Accepted values: `namespace`, `label:<key>`, or `annotation:<key>`
    #[arg(long, help_heading = "Service Discovery (Kubernetes)", value_parser = parse_model_id_from)]
    model_id_from: Option<String>,

    /// Worker discovery provider. Give this or `--service-discovery`, not both
    #[arg(
        long,
        value_enum,
        conflicts_with = "service_discovery",
        help_heading = "Service Discovery"
    )]
    discovery_provider: Option<Provider>,

    /// JSON manifest listing the workers, for `--discovery-provider file`. It
    /// may be created after startup
    #[arg(long, help_heading = "Service Discovery (File)")]
    discovery_file: Option<PathBuf>,

    /// Seconds between periodic discovery passes. Kubernetes, file and Consul
    /// discovery apply a change as it happens and use the pass to retry and
    /// to catch what went unreported; Slurm discovery polls at this interval
    /// (default: 30)
    #[arg(long, help_heading = "Service Discovery")]
    discovery_check_interval_secs: Option<u64>,

    /// Slurm job name whose nodes are regular workers, for
    /// `--discovery-provider slurm` (can be specified multiple times)
    #[arg(long, action = ArgAction::Append, help_heading = "Service Discovery (Slurm)")]
    slurm_job_name: Vec<String>,

    /// Slurm job name whose nodes are encode workers (can be specified
    /// multiple times)
    #[arg(long, action = ArgAction::Append, help_heading = "Service Discovery (Slurm)")]
    slurm_encode_job_name: Vec<String>,

    /// Slurm job name whose nodes are prefill workers (can be specified
    /// multiple times)
    #[arg(long, action = ArgAction::Append, help_heading = "Service Discovery (Slurm)")]
    slurm_prefill_job_name: Vec<String>,

    /// Slurm job name whose nodes are decode workers (can be specified
    /// multiple times)
    #[arg(long, action = ArgAction::Append, help_heading = "Service Discovery (Slurm)")]
    slurm_decode_job_name: Vec<String>,

    /// Port of the inference server on every node of a discovered job
    /// (required with `--discovery-provider slurm`)
    #[arg(long, help_heading = "Service Discovery (Slurm)")]
    slurm_port: Option<u16>,

    /// Bootstrap port of the Slurm-discovered encode and prefill workers
    #[arg(long, help_heading = "Service Discovery (Slurm)")]
    slurm_bootstrap_port: Option<u16>,

    /// Discover this Slurm user's jobs, by name or uid (default: the user the
    /// controller runs as)
    #[arg(long, help_heading = "Service Discovery (Slurm)")]
    slurm_user: Option<String>,

    /// Discover only jobs charged to this Slurm account
    #[arg(long, help_heading = "Service Discovery (Slurm)")]
    slurm_account: Option<String>,

    /// Discover only jobs in this Slurm partition
    #[arg(long, help_heading = "Service Discovery (Slurm)")]
    slurm_partition: Option<String>,

    /// Consul service whose instances with passing health checks are the
    /// workers (required with `--discovery-provider consul`). Each instance's
    /// `smg_*` service metadata configures its worker
    #[arg(long, help_heading = "Service Discovery (Consul)")]
    consul_service: Option<String>,

    /// Consul's HTTP API address (default: `CONSUL_HTTP_ADDR`, else
    /// 127.0.0.1:8500; HTTPS for one with no scheme when `CONSUL_HTTP_SSL` is
    /// true). The ACL token and TLS settings come from
    /// Consul's environment variables: `CONSUL_HTTP_TOKEN` or
    /// `CONSUL_HTTP_TOKEN_FILE`, `CONSUL_CACERT`, `CONSUL_CLIENT_CERT` and
    /// `CONSUL_CLIENT_KEY`
    #[arg(long, help_heading = "Service Discovery (Consul)")]
    consul_address: Option<String>,

    /// Consul datacenter to read (default: the Consul agent's own)
    #[arg(long, help_heading = "Service Discovery (Consul)")]
    consul_datacenter: Option<String>,
    /// Key to send to discovered workers (env: SMG_API_KEY). The flag beats
    /// the environment, and a key together with a key file is refused
    #[arg(long, value_parser = StringValueParser::new().map(Credential), help_heading = "Service Discovery")]
    api_key: Option<Credential>,

    /// File holding the worker key alone, such as a mounted secret (env:
    /// SMG_API_KEY_FILE). Surrounding whitespace is dropped
    #[arg(long, help_heading = "Service Discovery")]
    api_key_file: Option<PathBuf>,
}

/// The key a discovered worker is called with.
#[derive(Clone, Eq, PartialEq)]
pub(crate) struct Credential(String);

impl Credential {
    #[cfg(test)]
    pub(super) fn for_test(value: &str) -> Self {
        Self(value.to_owned())
    }

    pub(super) fn value(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Credential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<redacted>")
    }
}

impl fmt::Debug for DiscoveryFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiscoveryFlags")
            .field("provider", &self.provider())
            .field("api_key", &self.api_key)
            .finish_non_exhaustive()
    }
}

impl DiscoveryFlags {
    fn provider(&self) -> Option<Provider> {
        if self.service_discovery {
            Some(Provider::Kubernetes)
        } else {
            self.discovery_provider
        }
    }

    /// The selected provider with its roles and its key, or `None` where no
    /// flag selects one. The key's two variables are read through `env`, and
    /// only for a selected provider: a variable left set for another program
    /// fails no start that has no use for it.
    pub(crate) fn configured(
        &self,
        env: impl Fn(&str) -> Option<OsString>,
    ) -> Result<Option<DiscoverySettings>, ConfigError> {
        let Some(settings) = self.settings()? else {
            return Ok(None);
        };
        let credential = self.credential_with_env(env)?;
        DiscoverySettings::new(settings, self.roles(), credential).map(Some)
    }

    /// The selected provider's settings, or `None` where no flag selects one.
    fn settings(&self) -> Result<Option<DiscoveryConfig>, ConfigError> {
        let provider = self.provider();
        self.validate_companions(provider)?;
        let selector = labels("--selector", &self.selector)?;
        let encode_selector = labels("--encode-selector", &self.encode_selector)?;
        let prefill_selector = labels("--prefill-selector", &self.prefill_selector)?;
        let decode_selector = labels("--decode-selector", &self.decode_selector)?;

        let interval = self
            .discovery_check_interval_secs
            .unwrap_or(DEFAULT_DISCOVERY_CHECK_INTERVAL_SECS);
        if interval == 0 {
            return Err(refuse(
                "--discovery-check-interval-secs",
                "must be positive",
            ));
        }

        Ok(match provider {
            None => None,
            Some(Provider::Kubernetes) => {
                Some(DiscoveryConfig::Kubernetes(KubernetesDiscoveryConfig {
                    namespace: self.service_discovery_namespace.clone(),
                    port: self.service_discovery_port,
                    check_interval_secs: interval,
                    selector,
                    encode_selector,
                    prefill_selector,
                    decode_selector,
                    kv_connector_annotation: self.kv_connector_annotation.clone(),
                    kv_engine_id_annotation: self.kv_engine_id_annotation.clone(),
                    pairing_protocol_label: self.pairing_protocol_label.clone(),
                    model_id_source: self.model_id_from.clone(),
                    ..Default::default()
                }))
            }
            Some(Provider::File) => Some(DiscoveryConfig::File(FileDiscoveryConfig {
                path: self
                    .discovery_file
                    .as_ref()
                    .filter(|path| !path.as_os_str().is_empty())
                    .cloned()
                    .ok_or_else(|| refuse("--discovery-file", "give the manifest path"))?,
                check_interval_secs: interval,
            })),
            Some(Provider::Slurm) => {
                if !self.slurm_jobs().iter().any(|(_, names)| !names.is_empty()) {
                    return Err(refuse(
                        "--slurm-job-name",
                        "give at least one regular, encode, prefill or decode job name",
                    ));
                }
                Some(DiscoveryConfig::Slurm(SlurmDiscoveryConfig {
                    job_names: self.slurm_job_name.clone(),
                    encode_job_names: self.slurm_encode_job_name.clone(),
                    prefill_job_names: self.slurm_prefill_job_name.clone(),
                    decode_job_names: self.slurm_decode_job_name.clone(),
                    port: self
                        .slurm_port
                        .ok_or_else(|| refuse("--slurm-port", "give the worker port"))?,
                    bootstrap_port: self.slurm_bootstrap_port,
                    user: self.slurm_user.clone(),
                    account: self.slurm_account.clone(),
                    partition: self.slurm_partition.clone(),
                    check_interval_secs: interval,
                }))
            }
            Some(Provider::Consul) => Some(DiscoveryConfig::Consul(ConsulDiscoveryConfig {
                service: self
                    .consul_service
                    .as_ref()
                    .filter(|name| !name.is_empty())
                    .cloned()
                    .ok_or_else(|| refuse("--consul-service", "give the service name"))?,
                address: self.consul_address.clone(),
                datacenter: self.consul_datacenter.clone(),
                check_interval_secs: interval,
            })),
        })
    }

    fn slurm_jobs(&self) -> [(&'static str, &[String]); 4] {
        [
            ("--slurm-job-name", &self.slurm_job_name),
            ("--slurm-encode-job-name", &self.slurm_encode_job_name),
            ("--slurm-prefill-job-name", &self.slurm_prefill_job_name),
            ("--slurm-decode-job-name", &self.slurm_decode_job_name),
        ]
    }

    fn validate_companions(&self, provider: Option<Provider>) -> Result<(), ConfigError> {
        if provider != Some(Provider::File) && self.discovery_file.is_some() {
            return Err(refuse(
                "--discovery-file",
                "applies only to --discovery-provider file",
            ));
        }
        if provider.is_none() {
            // The key is sent to discovered workers alone.
            for (flag, given) in [
                (
                    "--discovery-check-interval-secs",
                    self.discovery_check_interval_secs.is_some(),
                ),
                ("--api-key", self.api_key.is_some()),
                ("--api-key-file", self.api_key_file.is_some()),
            ] {
                if given {
                    return Err(refuse(flag, "needs a discovery provider"));
                }
            }
        }
        if provider != Some(Provider::Slurm) {
            for (flag, given) in self
                .slurm_jobs()
                .map(|(flag, names)| (flag, !names.is_empty()))
                .into_iter()
                .chain([
                    ("--slurm-port", self.slurm_port.is_some()),
                    (
                        "--slurm-bootstrap-port",
                        self.slurm_bootstrap_port.is_some(),
                    ),
                    ("--slurm-user", self.slurm_user.is_some()),
                    ("--slurm-account", self.slurm_account.is_some()),
                    ("--slurm-partition", self.slurm_partition.is_some()),
                ])
            {
                if given {
                    return Err(refuse(flag, "applies only to --discovery-provider slurm"));
                }
            }
        }
        if provider != Some(Provider::Consul) {
            for (flag, given) in [
                ("--consul-service", self.consul_service.is_some()),
                ("--consul-address", self.consul_address.is_some()),
                ("--consul-datacenter", self.consul_datacenter.is_some()),
            ] {
                if given {
                    return Err(refuse(flag, "applies only to --discovery-provider consul"));
                }
            }
        }
        Ok(())
    }

    /// The roles the provider registers.
    fn roles(&self) -> Option<Roles> {
        let prefill_decode = [
            &self.prefill_selector,
            &self.decode_selector,
            &self.slurm_prefill_job_name,
            &self.slurm_decode_job_name,
        ]
        .into_iter()
        .any(|entries| !entries.is_empty());
        prefill_decode.then_some(
            if self.encode_selector.is_empty() && self.slurm_encode_job_name.is_empty() {
                Roles::PrefillDecode
            } else {
                Roles::EncodePrefillDecode
            },
        )
    }

    /// The credential the discovered workers are called with, if a flag or
    /// a variable read through `env` gives one.
    fn credential_with_env(
        &self,
        env: impl Fn(&str) -> Option<OsString>,
    ) -> Result<Option<Credential>, ConfigError> {
        let read_env = |name: &'static str, flag| {
            env(name)
                .map(|value| {
                    value.into_string().map_err(|_| {
                        refuse(flag, format!("{name} is set to a value that is not UTF-8"))
                    })
                })
                .transpose()
        };
        let key = match &self.api_key {
            Some(key) => Some(key.clone()),
            None => read_env(KEY_ENV, "--api-key")?.map(|key| Credential(key.trim().to_owned())),
        };
        let path = match &self.api_key_file {
            Some(path) => Some(path.clone()),
            None => read_env(KEY_FILE_ENV, "--api-key-file")?.map(PathBuf::from),
        };
        match (key, path) {
            (Some(_), Some(_)) => Err(refuse(
                "--api-key-file",
                format!("give one of --api-key ({KEY_ENV}) and --api-key-file ({KEY_FILE_ENV})"),
            )),
            (Some(key), None) if key.0.trim().is_empty() => {
                Err(refuse("--api-key", "the key is empty"))
            }
            (key, None) => Ok(key),
            (None, Some(path)) => read_key(&path).map(Some),
        }
    }
}

fn read_key(path: &Path) -> Result<Credential, ConfigError> {
    let bad_file = |reason| refuse("--api-key-file", format!("{}: {reason}", path.display()));
    let content = fs::read_to_string(path).map_err(|error| bad_file(error.to_string()))?;
    let value = content.trim();
    if value.is_empty() {
        return Err(bad_file("the file is empty".to_owned()));
    }
    if value.lines().count() > 1 {
        return Err(bad_file(
            "the file must hold the key alone on one line".to_owned(),
        ));
    }
    Ok(Credential(value.to_owned()))
}

fn labels(flag: &'static str, entries: &[String]) -> Result<HashMap<String, String>, ConfigError> {
    let mut labels = HashMap::new();
    let invalid = || refuse(flag, "expected 'key=value' selector entries");
    for entry in entries {
        let mut tokens = entry.split_whitespace().peekable();
        if tokens.peek().is_none() {
            return Err(invalid());
        }
        for token in tokens {
            let Some((key, value)) = token.split_once('=').filter(|(key, _)| !key.is_empty())
            else {
                return Err(invalid());
            };
            labels.insert(key.to_owned(), value.to_owned());
        }
    }
    Ok(labels)
}

fn parse_model_id_from(value: &str) -> Result<String, String> {
    ModelIdSource::parse(value)?;
    Ok(value.to_owned())
}

/// The refusal of settings the discovery crate refuses, under the flag
/// that sets the refused field. A refusal that names no field stands under
/// the provider's flag.
pub(super) fn refuse_settings(
    config: &DiscoveryConfig,
    roles: Roles,
    error: DiscoveryConfigError,
) -> ConfigError {
    const PROVIDER_FLAG: &str = "--discovery-provider";
    const ROLES: &str =
        "the roles follow the prefill, decode and encode selectors and job names given";
    let DiscoveryConfigError::InvalidValue { field, .. } = &error else {
        // The settings' own check names no field only where it refuses the
        // roles, in the words of a host that has a routing mode. The
        // controller has none, so it says where its roles come from. A
        // refusal from elsewhere, such as a provider that fails to build,
        // keeps its own words.
        let words = error.to_string();
        let reason = match config.validate(roles) {
            Err(check) if check.to_string() == words => format!("{words}; {ROLES}"),
            _ => words,
        };
        return refuse(PROVIDER_FLAG, reason);
    };
    let flag = match (config, field.as_str()) {
        (_, "discovery.check_interval_secs") => "--discovery-check-interval-secs",
        (DiscoveryConfig::Kubernetes(_), "discovery.port") => "--service-discovery-port",
        (DiscoveryConfig::Kubernetes(_), "discovery.kv_connector_annotation") => {
            "--kv-connector-annotation"
        }
        (DiscoveryConfig::Kubernetes(_), "discovery.kv_engine_id_annotation") => {
            "--kv-engine-id-annotation"
        }
        (DiscoveryConfig::Kubernetes(_), "discovery.pairing_protocol_label") => {
            "--pairing-protocol-label"
        }
        (DiscoveryConfig::Kubernetes(_), "discovery.model_id_source") => "--model-id-from",
        (DiscoveryConfig::File(_), "discovery.path") => "--discovery-file",
        (DiscoveryConfig::Slurm(_), "discovery.port") => "--slurm-port",
        (DiscoveryConfig::Slurm(_), "discovery.bootstrap_port") => "--slurm-bootstrap-port",
        (DiscoveryConfig::Slurm(_), "discovery.job_names") => "--slurm-job-name",
        (DiscoveryConfig::Slurm(_), "discovery.encode_job_names") => "--slurm-encode-job-name",
        (DiscoveryConfig::Slurm(_), "discovery.prefill_job_names") => "--slurm-prefill-job-name",
        (DiscoveryConfig::Slurm(_), "discovery.decode_job_names") => "--slurm-decode-job-name",
        (DiscoveryConfig::Slurm(_), "discovery.user") => "--slurm-user",
        (DiscoveryConfig::Slurm(_), "discovery.account") => "--slurm-account",
        (DiscoveryConfig::Slurm(_), "discovery.partition") => "--slurm-partition",
        (DiscoveryConfig::Consul(_), "discovery.service") => "--consul-service",
        (DiscoveryConfig::Consul(_), "discovery.address") => "--consul-address",
        (DiscoveryConfig::Consul(_), "discovery.datacenter") => "--consul-datacenter",
        _ => PROVIDER_FLAG,
    };
    refuse(flag, error.to_string())
}

fn refuse(flag: &'static str, reason: impl Into<String>) -> ConfigError {
    ConfigError::DiscoveryFlag {
        flag,
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests;
