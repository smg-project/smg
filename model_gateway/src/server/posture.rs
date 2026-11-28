//! The gateway's authentication posture, named once at startup.
//!
//! Nothing is authenticated unless a key is configured: the admin and
//! worker-management routes, the serving routes and the metrics listener
//! all answer whoever can reach them. That default keeps a local start to
//! one command, but a deployment that reaches it by omission should learn
//! so from its first log lines, not from an audit. One WARN record names
//! every open surface and the flag that closes it.
//!
//! A keyed start names where its secrets came from instead: the command
//! line, the environment or a file. One INFO record carries the sources;
//! no record carries a value.

use std::net::SocketAddr;

use smg_mesh::MeshServerConfig;
use tracing::{info, warn};

use crate::{middleware::AuthConfig, observability::metrics::PrometheusConfig};

/// Where a secret (a serving key, a store URL) was read from at start-up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecretSource {
    /// Given on the command line.
    Flag,
    /// Read from an environment variable.
    Environment,
    /// Read from a file the command line or the environment named.
    File,
    /// Given as a value by the launcher that started this gateway, from
    /// its flag or its environment, which the binding does not see.
    Launcher,
}

impl SecretSource {
    /// The one word the start-up record uses.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Flag => "flag",
            Self::Environment => "env",
            Self::File => "file",
            Self::Launcher => "launcher",
        }
    }
}

/// The sources of the secrets a start was given, for the start-up record;
/// `None` is a secret that was not given. The values are not here: the
/// record names sources alone.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SecretSources {
    /// The shared serving API key.
    pub api_key: Option<SecretSource>,
    /// The per-tenant serving keys, with how many entries the source held.
    pub tenant_api_keys: Option<(SecretSource, usize)>,
    /// The control-plane API keys, with how many entries the source held.
    pub control_plane_api_keys: Option<(SecretSource, usize)>,
    /// The PostgreSQL connection URL.
    pub postgres_db_url: Option<SecretSource>,
    /// The Redis connection URL.
    pub redis_url: Option<SecretSource>,
}

impl SecretSources {
    /// Whether any secret was given.
    pub fn any(&self) -> bool {
        self.api_key.is_some()
            || self.tenant_api_keys.is_some()
            || self.control_plane_api_keys.is_some()
            || self.postgres_db_url.is_some()
            || self.redis_url.is_some()
    }
}

/// The headline of the sources record.
pub(crate) const SOURCES_HEADLINE: &str =
    "CREDENTIAL SOURCES: where this start read its secrets (the values are never logged)";

/// Logs the sources record, one INFO, when a secret was given: one field per
/// secret naming its source, `flag`, `env`, `file` or `launcher`, and whether the key
/// files are watched for changes (`secrets_auto_reload`: `off`, or `on` with
/// the names of the settings whose files are watched).
pub(crate) fn log_secret_sources(sources: &SecretSources, secrets_auto_reload: &str) {
    if !sources.any() {
        return;
    }
    let entries_of = |source: Option<(SecretSource, usize)>| {
        source.map(|(source, entries)| format!("{} ({entries} entries)", source.as_str()))
    };
    let tenant_api_keys = entries_of(sources.tenant_api_keys);
    let control_plane_api_keys = entries_of(sources.control_plane_api_keys);
    info!(
        api_key = sources.api_key.map(SecretSource::as_str),
        tenant_api_keys = tenant_api_keys.as_deref(),
        control_plane_api_keys = control_plane_api_keys.as_deref(),
        postgres_db_url = sources.postgres_db_url.map(SecretSource::as_str),
        redis_url = sources.redis_url.map(SecretSource::as_str),
        secrets_auto_reload,
        "{SOURCES_HEADLINE}"
    );
}

/// The headline of the gated-routes record.
pub(crate) const GATED_HEADLINE: &str =
    "GATED CONTROL-PLANE ROUTES: mounted only behind control-plane authentication";

/// Logs the gated-routes record, one INFO: whether the routes that exist
/// only behind control-plane authentication are mounted on this start,
/// and which they are, so an operator reading a 404 for one of them
/// finds the reason in the start-up log.
pub(crate) fn log_gated_routes(control_plane_auth: bool, routes: &[(&str, &str)]) {
    let routes = routes
        .iter()
        .map(|(method, path)| format!("{method} {path}"))
        .collect::<Vec<_>>()
        .join(", ");
    if control_plane_auth {
        info!(mounted = true, routes, "{GATED_HEADLINE}");
    } else {
        info!(
            mounted = false,
            routes,
            "{GATED_HEADLINE}; this start has none, so they answer 404; configure \
             --control-plane-api-keys, --control-plane-api-keys-file or --jwt-issuer/--jwt-audience to mount them"
        );
    }
}

/// The planes a start leaves open, judged the way `build_app` guards them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OpenSurfaces {
    /// The admin and worker-management routes answer without a credential:
    /// no control-plane auth and no shared `--api-key` to fall back to.
    /// Tenant keys alone close them (every request is denied), so they do
    /// not count as open.
    pub control_plane: bool,
    /// The serving routes answer without a credential.
    pub data_plane: bool,
    /// The mesh listener, when the mesh runs without TLS: it gossips
    /// membership and state with any peer that connects.
    pub mesh_without_tls: Option<SocketAddr>,
}

impl OpenSurfaces {
    /// `control_plane_auth` says whether control-plane auth (API keys or
    /// JWT) initialised; `serving_auth` and `admin_auth` are the serving
    /// check and the admin routes' shared-key fallback; `mesh` is the mesh
    /// configuration of this start, if any.
    pub(crate) fn judge(
        serving_auth: &AuthConfig,
        admin_auth: &AuthConfig,
        control_plane_auth: bool,
        mesh: Option<&MeshServerConfig>,
    ) -> Self {
        // Mirrors `build_app`: control-plane auth guards the admin routes;
        // without it, tenant keys without a shared key deny every request;
        // otherwise the shared-key check runs, which passes everything when
        // there is no key at all.
        let control_plane =
            !control_plane_auth && !admin_auth.is_enabled() && !serving_auth.is_enabled();
        Self {
            control_plane,
            data_plane: !serving_auth.is_enabled(),
            mesh_without_tls: mesh
                .filter(|mesh| mesh.mtls_config.is_none())
                .map(|mesh| mesh.bind_addr),
        }
    }

    pub(crate) fn any(self) -> bool {
        self.control_plane || self.data_plane || self.mesh_without_tls.is_some()
    }
}

/// The headline of the record; the fields name the surfaces.
pub(crate) const HEADLINE: &str = "SECURITY POSTURE: this gateway runs without authentication; \
                                  every surface named here answers any caller that can reach it";

/// The warning for an open start: one field per open surface, each naming
/// the routes and the flag that closes them. `None` when both planes are
/// keyed. The metrics listener has no authentication of its own, so it is
/// named whenever the record is emitted, with the binding to narrow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PostureWarning {
    pub control_plane: Option<String>,
    pub data_plane: Option<String>,
    pub metrics_listener: Option<String>,
}

pub(crate) fn open_posture_warning(
    open: OpenSurfaces,
    metrics: Option<&PrometheusConfig>,
    admin_listener: Option<SocketAddr>,
) -> Option<PostureWarning> {
    if !open.any() {
        return None;
    }
    let mut control_plane_parts = Vec::new();
    if open.control_plane {
        // With an admin listener the routes answer there alone: the record
        // names it, so the address to fence is the one in the record.
        let listener = admin_listener.map_or_else(String::new, |addr| {
            format!(" (on the admin listener {addr})")
        });
        control_plane_parts.push(format!(
            "OPEN{listener}: POST /workers and PUT/PATCH/DELETE /workers/{{id}} (register or remove \
             backends by URL), GET /workers and GET /get_loads (the backends' addresses), POST \
             /flush_cache (drops the engines' prefix caches), POST /start_profile and POST \
             /stop_profile (the engines' profilers), POST /heap_profile (writes this gateway's \
             heap profile to disk), /parse/*, /wasm and /v1/tokenizers; protect them with \
             --control-plane-api-keys id:name:admin:<key> or --jwt-issuer/--jwt-audience (the \
             shared --api-key gates them too)"
        ));
    }
    if let Some(addr) = open.mesh_without_tls {
        control_plane_parts.push(format!(
            "OPEN: the mesh listener on {addr} gossips membership and state with any peer that \
             connects, in clear text; give it --mesh-tls-ca-cert, --mesh-tls-cert and \
             --mesh-tls-key (peers then need a certificate from that CA)"
        ));
    }
    let control_plane = (!control_plane_parts.is_empty()).then(|| control_plane_parts.join(" "));
    let data_plane = open.data_plane.then(|| {
        "OPEN: /v1/chat/completions, /v1/completions, /v1/responses, /v1/embeddings, /v1/messages \
         and the other serving routes need no credential; protect them with --api-key <key> or \
         --tenant-api-key tenant:<key>"
            .to_string()
    });
    let metrics_listener = metrics.map(|metrics| {
        // An IPv6 host is bracketed so the address reads as one token.
        let host = if metrics.host.contains(':') {
            format!("[{}]", metrics.host)
        } else {
            metrics.host.clone()
        };
        format!(
            "{host}:{} serves /metrics (per-worker series named by backend address) with no \
             authentication; bind it to a private address with --prometheus-host and fence the \
             port with a network policy",
            metrics.port
        )
    });
    Some(PostureWarning {
        control_plane,
        data_plane,
        metrics_listener,
    })
}

/// Logs the record, one WARN, when a plane is open. `admin_listener` is the
/// admin listener's address when the control plane has one.
pub(crate) fn log_open_posture(
    serving_auth: &AuthConfig,
    admin_auth: &AuthConfig,
    control_plane_auth: bool,
    metrics: Option<&PrometheusConfig>,
    mesh: Option<&MeshServerConfig>,
    admin_listener: Option<SocketAddr>,
) {
    let open = OpenSurfaces::judge(serving_auth, admin_auth, control_plane_auth, mesh);
    if let Some(warning) = open_posture_warning(open, metrics, admin_listener) {
        warn!(
            control_plane = warning.control_plane.as_deref(),
            data_plane = warning.data_plane.as_deref(),
            metrics_listener = warning.metrics_listener.as_deref(),
            "{HEADLINE}"
        );
    }
}

#[cfg(test)]
#[path = "../../tests/unit/server/posture/tests.rs"]
mod tests;
