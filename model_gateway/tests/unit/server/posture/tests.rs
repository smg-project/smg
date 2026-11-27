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
    let open = OpenSurfaces::judge(&unkeyed(), &unkeyed(), false, None);
    assert_eq!(
        open,
        OpenSurfaces {
            control_plane: true,
            data_plane: true,
            mesh_without_tls: None,
        }
    );

    let warning = open_posture_warning(open, Some(&listener()), None).expect("an open start warns");
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

/// With an admin listener, the open control plane is named with the
/// listener's address: that is the address to fence.
#[test]
fn an_admin_listener_is_named_with_the_open_control_plane() {
    let open = OpenSurfaces::judge(&unkeyed(), &unkeyed(), false, None);
    let admin: SocketAddr = "10.0.0.5:9001".parse().unwrap();
    let warning = open_posture_warning(open, None, Some(admin)).expect("an open start warns");
    let control_plane = warning.control_plane.expect("the control plane is named");
    assert!(
        control_plane.starts_with("OPEN (on the admin listener 10.0.0.5:9001): POST /workers"),
        "{control_plane}"
    );
    assert!(warning.data_plane.is_some());

    let without = open_posture_warning(open, None, None).expect("an open start warns");
    assert!(
        without
            .control_plane
            .expect("named")
            .starts_with("OPEN: POST /workers"),
        "no listener named without one"
    );
}

#[test]
fn control_plane_keys_alone_leave_only_the_data_plane_open() {
    let open = OpenSurfaces::judge(&unkeyed(), &unkeyed(), true, None);
    assert_eq!(
        open,
        OpenSurfaces {
            control_plane: false,
            data_plane: true,
            mesh_without_tls: None,
        }
    );

    let warning = open_posture_warning(open, None, None).expect("the data plane is open");

    assert!(warning.data_plane.is_some());
    assert_eq!(warning.control_plane, None);
    assert_eq!(warning.metrics_listener, None);
}

#[test]
fn an_ipv6_listener_is_bracketed() {
    let open = OpenSurfaces::judge(&unkeyed(), &unkeyed(), false, None);
    let listener = PrometheusConfig {
        host: "::".to_string(),
        ..PrometheusConfig::default()
    };

    let warning = open_posture_warning(open, Some(&listener), None).expect("an open start warns");

    assert!(warning
        .metrics_listener
        .as_deref()
        .is_some_and(|line| line.starts_with("[::]:29000 ")));
}

#[test]
fn a_shared_key_closes_both_planes() {
    let open = OpenSurfaces::judge(&keyed(), &keyed(), false, None);

    assert!(!open.any());
    assert_eq!(open_posture_warning(open, Some(&listener()), None), None);
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

    let open = OpenSurfaces::judge(&serving, &unkeyed(), false, None);

    assert!(!open.any());
}

#[traced_test]
#[test]
fn an_open_start_logs_the_warning() {
    log_open_posture(&unkeyed(), &unkeyed(), false, Some(&listener()), None, None);

    assert!(logs_contain("SECURITY POSTURE"));
    assert!(logs_contain("control_plane=\"OPEN: POST /workers"));
    assert!(logs_contain("data_plane=\"OPEN: /v1/chat/completions"));
    assert!(logs_contain("metrics_listener=\"0.0.0.0:29000"));
}

fn mesh(mtls: Option<smg_mesh::MTLSConfig>) -> MeshServerConfig {
    let addr: SocketAddr = "10.0.0.7:39527".parse().unwrap();
    MeshServerConfig {
        self_name: "node-a".to_string(),
        bind_addr: addr,
        advertise_addr: addr,
        init_peers: Vec::new(),
        mtls_config: mtls,
    }
}

/// A mesh without TLS is an open control-plane surface even on a keyed
/// start; with mTLS configured it is not named.
#[test]
fn a_mesh_listener_without_tls_is_named_in_the_control_plane() {
    let open = OpenSurfaces::judge(&keyed(), &keyed(), true, Some(&mesh(None)));
    assert_eq!(
        open.mesh_without_tls,
        Some("10.0.0.7:39527".parse().unwrap())
    );
    assert!(open.any());

    let warning = open_posture_warning(open, None, None).expect("a plaintext mesh warns");
    let control_plane = warning
        .control_plane
        .expect("the mesh is a control-plane surface");
    for needle in [
        "mesh listener on 10.0.0.7:39527",
        "--mesh-tls-ca-cert",
        "--mesh-tls-key",
    ] {
        assert!(
            control_plane.contains(needle),
            "{needle} missing from:\n{control_plane}"
        );
    }
    assert!(
        !control_plane.contains("POST /workers"),
        "the keyed routes are not named"
    );
    assert_eq!(warning.data_plane, None);

    let secured = OpenSurfaces::judge(
        &keyed(),
        &keyed(),
        true,
        Some(&mesh(Some(smg_mesh::MTLSConfig::default()))),
    );
    assert_eq!(secured.mesh_without_tls, None);
    assert!(!secured.any());
}

#[traced_test]
#[test]
fn a_plaintext_mesh_logs_the_warning_on_a_keyed_start() {
    log_open_posture(
        &keyed(),
        &keyed(),
        true,
        Some(&listener()),
        Some(&mesh(None)),
        None,
    );

    assert!(logs_contain("SECURITY POSTURE"));
    assert!(logs_contain("mesh listener on 10.0.0.7:39527"));
}

#[traced_test]
#[test]
fn a_keyed_start_logs_nothing() {
    log_open_posture(&keyed(), &keyed(), true, Some(&listener()), None, None);

    assert!(!logs_contain("SECURITY POSTURE"));
}

/// The sources record names where each secret came from and how many
/// entries a tenant source held; a secret that was not given has no
/// field. The record never sees a value, so it cannot carry one.
#[traced_test]
#[test]
fn the_sources_record_names_sources_alone() {
    log_secret_sources(
        &SecretSources {
            api_key: Some(SecretSource::File),
            tenant_api_keys: Some((SecretSource::Environment, 3)),
            control_plane_api_keys: Some((SecretSource::File, 2)),
            postgres_db_url: Some(SecretSource::Flag),
            redis_url: None,
        },
        "on (api-key-file, control-plane-api-keys-file)",
    );

    assert!(logs_contain("CREDENTIAL SOURCES"));
    assert!(logs_contain(
        "secrets_auto_reload=\"on (api-key-file, control-plane-api-keys-file)\""
    ));
    assert!(logs_contain("api_key=\"file\""));
    assert!(logs_contain("tenant_api_keys=\"env (3 entries)\""));
    assert!(logs_contain("control_plane_api_keys=\"file (2 entries)\""));
    assert!(logs_contain("postgres_db_url=\"flag\""));
    assert!(!logs_contain("redis_url="));
}

/// The launcher's word in the record, beside the command line's three.
#[test]
fn a_launcher_given_secret_reads_launcher() {
    assert_eq!(SecretSource::Launcher.as_str(), "launcher");
    assert_eq!(SecretSource::Flag.as_str(), "flag");
}

/// The gated-routes record says whether the routes are mounted and
/// names them in both states, with the flags that mount them when
/// they are off.
#[traced_test]
#[test]
fn the_gated_routes_record_names_the_routes_in_both_states() {
    log_gated_routes(false, &[("GET", "/config")]);
    assert!(logs_contain("GATED CONTROL-PLANE ROUTES"));
    assert!(logs_contain("mounted=false"));
    assert!(logs_contain("routes=\"GET /config\""));
    assert!(logs_contain("--control-plane-api-keys"));

    log_gated_routes(true, &[("GET", "/config"), ("PUT", "/rate-limits")]);
    assert!(logs_contain("mounted=true"));
    assert!(logs_contain("routes=\"GET /config, PUT /rate-limits\""));
}

#[traced_test]
#[test]
fn a_start_without_secrets_logs_no_sources_record() {
    assert!(!SecretSources::default().any());

    log_secret_sources(&SecretSources::default(), "off");

    assert!(!logs_contain("CREDENTIAL SOURCES"));
}
