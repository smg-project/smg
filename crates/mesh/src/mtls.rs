//! mTLS (mutual TLS) support for mesh cluster communication
//!
//! Provides optional mTLS encryption for gRPC mesh connections using rustls.

use std::path::PathBuf;

use anyhow::Result;
use tokio::{fs, sync::OnceCell};
use tonic::transport::{Certificate, ClientTlsConfig, Identity, ServerTlsConfig};

/// mTLS configuration
#[derive(Debug, Clone)]
pub struct MTLSConfig {
    /// Path to CA certificate file
    pub ca_cert_path: PathBuf,
    /// Path to server certificate file
    pub server_cert_path: PathBuf,
    /// Path to server private key file
    pub server_key_path: PathBuf,
    /// Whether to require client certificates
    pub require_client_cert: bool,
}

impl Default for MTLSConfig {
    fn default() -> Self {
        Self {
            ca_cert_path: PathBuf::from("/etc/ssl/certs/ca-certificates.crt"),
            server_cert_path: PathBuf::from("/etc/ssl/certs/server.crt"),
            server_key_path: PathBuf::from("/etc/ssl/private/server.key"),
            require_client_cert: true,
        }
    }
}

/// The TLS server name for a peer dialled at `host`, the host of its URL.
///
/// An IPv6 literal comes out of a URL bracketed (`[fd00::1]`), which is
/// neither a DNS name nor an IP address to the TLS stack; without the
/// brackets it is an IP server name and the peer's certificate is checked
/// for that IP SAN. IPv4 literals (an IP server name too) and DNS names
/// pass through unchanged.
pub(crate) fn tls_server_name(host: &str) -> &str {
    host.strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .unwrap_or(host)
}

/// The node's TLS material: the paths, and what they hold once read.
#[derive(Debug)]
pub struct MTLSManager {
    config: MTLSConfig,
    /// The CA and this node's identity, read from their files the first time
    /// the listener or a dial asks for them and kept for the life of the
    /// process: the probes dial once a second, and read the three files (the
    /// CA twice) on every dial before.
    material: OnceCell<Material>,
}

/// What the three files hold: the CA the peers' certificates chain to and
/// this node's certificate with its key.
#[derive(Debug)]
struct Material {
    ca: Certificate,
    identity: Identity,
}

impl MTLSManager {
    /// Create a new mTLS manager
    pub fn new(config: MTLSConfig) -> Self {
        Self {
            config,
            material: OnceCell::new(),
        }
    }

    /// The material, read on the first call. A read that fails leaves the
    /// cell empty, so the next call reads again.
    async fn material(&self) -> Result<&Material> {
        self.material
            .get_or_try_init(|| async {
                let ca = fs::read(&self.config.ca_cert_path).await?;
                let cert = fs::read(&self.config.server_cert_path).await?;
                let key = fs::read(&self.config.server_key_path).await?;
                Ok(Material {
                    ca: Certificate::from_pem(ca),
                    identity: Identity::from_pem(cert, key),
                })
            })
            .await
    }

    /// What the mesh listener serves: this node's certificate and key, and,
    /// when client certificates are required, the CA as the trust root the
    /// peers' certificates must chain to.
    pub async fn server_tls_config(&self) -> Result<ServerTlsConfig> {
        let material = self.material().await?;
        let mut tls = ServerTlsConfig::new().identity(material.identity.clone());
        if self.config.require_client_cert {
            tls = tls.client_ca_root(material.ca.clone());
        }
        Ok(tls)
    }

    /// What a dial to the peer named `server_name` trusts and presents: the
    /// CA as the trust root, this node's certificate as its identity.
    pub async fn client_tls_config(&self, server_name: &str) -> Result<ClientTlsConfig> {
        let material = self.material().await?;
        Ok(ClientTlsConfig::new()
            .domain_name(server_name)
            .ca_certificate(material.ca.clone())
            .identity(material.identity.clone()))
    }
}

#[cfg(test)]
mod tests {
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
}
