use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

use tonic::transport::{ClientTlsConfig, Endpoint};

use super::{tls_server_name, MTLSConfig, MTLSManager};

#[test]
fn an_ipv6_peer_url_names_the_server_by_its_address() {
    let endpoint = Endpoint::from_static("https://[fd00::1]:39527");
    let host = endpoint.uri().host().unwrap();
    assert_eq!(host, "[fd00::1]", "the URL host keeps the brackets");

    let name = tls_server_name(host);

    assert_eq!(name, "fd00::1");
    assert_eq!(
        name.parse::<IpAddr>().unwrap(),
        IpAddr::V6(Ipv6Addr::new(0xfd00, 0, 0, 0, 0, 0, 0, 1))
    );
}

#[test]
fn an_ipv4_peer_url_names_the_server_by_its_address() {
    let endpoint = Endpoint::from_static("https://10.0.0.7:39527");

    let name = tls_server_name(endpoint.uri().host().unwrap());

    assert_eq!(name, "10.0.0.7");
    assert_eq!(
        name.parse::<IpAddr>().unwrap(),
        IpAddr::V4(Ipv4Addr::new(10, 0, 0, 7))
    );
}

#[test]
fn a_dns_peer_url_names_the_server_by_its_name() {
    let endpoint = Endpoint::from_static("https://mesh-1.mesh.svc:39527");

    assert_eq!(
        tls_server_name(endpoint.uri().host().unwrap()),
        "mesh-1.mesh.svc"
    );
}

#[test]
fn the_tls_stack_takes_the_unbracketed_name_only() {
    let bracketed = Endpoint::from_static("https://[::1]:1")
        .tls_config(ClientTlsConfig::new().domain_name("[::1]"));
    assert!(bracketed.is_err(), "a bracketed literal is no server name");

    let unbracketed = Endpoint::from_static("https://[::1]:1")
        .tls_config(ClientTlsConfig::new().domain_name(tls_server_name("[::1]")));
    assert!(unbracketed.is_ok(), "{:?}", unbracketed.err());
}

/// The material is read once: a manager that served a configuration
/// keeps serving it after its files are gone, so the dials of a member
/// (a probe a second) never read the disk again.
#[tokio::test]
async fn the_material_is_read_once_and_serves_after_its_files_are_gone() {
    let dir = tempfile::tempdir().unwrap();
    let file = |name: &str, bytes: &str| {
        let path = dir.path().join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    };
    let manager = MTLSManager::new(MTLSConfig {
        ca_cert_path: file("ca.pem", "ca"),
        server_cert_path: file("cert.pem", "cert"),
        server_key_path: file("key.pem", "key"),
        require_client_cert: true,
    });
    assert!(manager.client_tls_config("peer").await.is_ok());

    drop(dir);

    assert!(
        manager.client_tls_config("peer").await.is_ok(),
        "the client configuration comes from the material read at first"
    );
    assert!(
        manager.server_tls_config().await.is_ok(),
        "the server configuration comes from the same material"
    );
}
