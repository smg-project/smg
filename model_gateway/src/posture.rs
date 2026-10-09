//! The gateway's authentication posture, named once at startup.
//!
//! Nothing is authenticated unless a key is configured: the admin and
//! worker-management routes, the serving routes and the metrics listener
//! all answer whoever can reach them. That default keeps a local start to
//! one command, but a deployment that reaches it by omission should learn
//! so from its first log lines, not from an audit. One WARN record names
//! every open surface and the flag that closes it.

use tracing::warn;

use crate::{middleware::AuthConfig, observability::metrics::PrometheusConfig};

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
}

impl OpenSurfaces {
    /// `control_plane_auth` says whether control-plane auth (API keys or
    /// JWT) initialised; `serving_auth` and `admin_auth` are the serving
    /// check and the admin routes' shared-key fallback.
    pub(crate) fn judge(
        serving_auth: &AuthConfig,
        admin_auth: &AuthConfig,
        control_plane_auth: bool,
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
        }
    }

    pub(crate) fn any(self) -> bool {
        self.control_plane || self.data_plane
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
) -> Option<PostureWarning> {
    if !open.any() {
        return None;
    }
    let control_plane = open.control_plane.then(|| {
        "OPEN: POST /workers and PUT/PATCH/DELETE /workers/{id} (register or remove backends by \
         URL), GET /workers and GET /get_loads (the backends' addresses), POST /flush_cache (drops \
         the engines' prefix caches), POST /start_profile and POST /stop_profile (the engines' \
         profilers), POST /heap_profile (writes this gateway's heap profile to disk), /parse/*, \
         /wasm and /v1/tokenizers; protect them with --control-plane-api-keys \
         id:name:admin:<key> or --jwt-issuer/--jwt-audience (the shared --api-key gates them too)"
            .to_string()
    });
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

/// Logs the record, one WARN, when a plane is open.
pub(crate) fn log_open_posture(
    serving_auth: &AuthConfig,
    admin_auth: &AuthConfig,
    control_plane_auth: bool,
    metrics: Option<&PrometheusConfig>,
) {
    let open = OpenSurfaces::judge(serving_auth, admin_auth, control_plane_auth);
    if let Some(warning) = open_posture_warning(open, metrics) {
        warn!(
            control_plane = warning.control_plane.as_deref(),
            data_plane = warning.data_plane.as_deref(),
            metrics_listener = warning.metrics_listener.as_deref(),
            "{HEADLINE}"
        );
    }
}

#[cfg(test)]
mod tests {
    use tracing_test::traced_test;

    use super::*;
    use crate::config::TenantApiKeyEntry;

    fn unkeyed() -> AuthConfig {
        AuthConfig::new(None)
    }

    fn keyed() -> AuthConfig {
        AuthConfig::new(Some("test-key".to_string()))
    }

    fn listener() -> PrometheusConfig {
        PrometheusConfig::default()
    }

    #[test]
    fn an_unkeyed_start_names_every_open_surface_and_its_flag() {
        let open = OpenSurfaces::judge(&unkeyed(), &unkeyed(), false);
        assert_eq!(
            open,
            OpenSurfaces {
                control_plane: true,
                data_plane: true,
            }
        );

        let warning = open_posture_warning(open, Some(&listener())).expect("an open start warns");
        let message = format!(
            "{HEADLINE} {} {} {}",
            warning.control_plane.as_deref().unwrap_or_default(),
            warning.data_plane.as_deref().unwrap_or_default(),
            warning.metrics_listener.as_deref().unwrap_or_default()
        );

        for needle in [
            "SECURITY POSTURE",
            "POST /workers",
            "DELETE /workers/{id}",
            "/flush_cache",
            "/start_profile",
            "/stop_profile",
            "/heap_profile",
            "--control-plane-api-keys",
            "--jwt-issuer",
            "/v1/chat/completions",
            "--api-key",
            "--tenant-api-key",
            "0.0.0.0:29000",
            "--prometheus-host",
        ] {
            assert!(
                message.contains(needle),
                "{needle} missing from:\n{message}"
            );
        }
    }

    #[test]
    fn control_plane_keys_alone_leave_only_the_data_plane_open() {
        let open = OpenSurfaces::judge(&unkeyed(), &unkeyed(), true);
        assert_eq!(
            open,
            OpenSurfaces {
                control_plane: false,
                data_plane: true,
            }
        );

        let warning = open_posture_warning(open, None).expect("the data plane is open");

        assert!(warning.data_plane.is_some());
        assert_eq!(warning.control_plane, None);
        assert_eq!(warning.metrics_listener, None);
    }

    #[test]
    fn an_ipv6_listener_is_bracketed() {
        let open = OpenSurfaces::judge(&unkeyed(), &unkeyed(), false);
        let listener = PrometheusConfig {
            host: "::".to_string(),
            ..PrometheusConfig::default()
        };

        let warning = open_posture_warning(open, Some(&listener)).expect("an open start warns");

        assert!(warning
            .metrics_listener
            .as_deref()
            .is_some_and(|line| line.starts_with("[::]:29000 ")));
    }

    #[test]
    fn a_shared_key_closes_both_planes() {
        let open = OpenSurfaces::judge(&keyed(), &keyed(), false);

        assert!(!open.any());
        assert_eq!(open_posture_warning(open, Some(&listener())), None);
    }

    /// Tenant keys gate the serving routes, and without a shared key the
    /// admin routes deny every request: closed, not open.
    #[test]
    fn tenant_keys_alone_close_both_planes() {
        let serving = AuthConfig::with_tenant_keys(
            None,
            &[TenantApiKeyEntry {
                tenant_id: "team-red".to_string(),
                key: "test-tenant-key".to_string(),
            }],
        );

        let open = OpenSurfaces::judge(&serving, &unkeyed(), false);

        assert!(!open.any());
    }

    #[traced_test]
    #[test]
    fn an_open_start_logs_the_warning() {
        log_open_posture(&unkeyed(), &unkeyed(), false, Some(&listener()));

        assert!(logs_contain("SECURITY POSTURE"));
        assert!(logs_contain("control_plane=\"OPEN: POST /workers"));
        assert!(logs_contain("data_plane=\"OPEN: /v1/chat/completions"));
        assert!(logs_contain("metrics_listener=\"0.0.0.0:29000"));
    }

    #[traced_test]
    #[test]
    fn a_keyed_start_logs_nothing() {
        log_open_posture(&keyed(), &keyed(), true, Some(&listener()));

        assert!(!logs_contain("SECURITY POSTURE"));
    }
}
