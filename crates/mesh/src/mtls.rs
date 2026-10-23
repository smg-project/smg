//! mTLS (mutual TLS) support for mesh cluster communication
//!
//! Provides optional mTLS encryption for gRPC mesh connections using rustls.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::Result;
use rustls::{pki_types::CertificateDer, ClientConfig, RootCertStore};
use rustls_pemfile::certs;
use tokio::fs;
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

/// mTLS certificate manager
#[derive(Debug)]
pub struct MTLSManager {
    config: MTLSConfig,
}

impl MTLSManager {
    /// Create a new mTLS manager
    pub fn new(config: MTLSConfig) -> Self {
        Self { config }
    }

    /// Load client TLS configuration
    pub async fn load_client_config(&self) -> Result<Arc<ClientConfig>> {
        let mut root_store = RootCertStore::empty();

        // Load CA certificate
        let ca_certs = self.load_certs(&self.config.ca_cert_path).await?;
        for cert in ca_certs {
            root_store.add(cert)?;
        }

        let mut client_config = ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_no_client_auth();

        // Enable ALPN for HTTP/2
        client_config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];

        Ok(Arc::new(client_config))
    }

    /// Load CA certificate for tonic client TLS configuration
    pub async fn load_ca_certificate(&self) -> Result<Certificate> {
        let ca_cert = fs::read(&self.config.ca_cert_path).await?;
        Ok(Certificate::from_pem(ca_cert))
    }

    /// This node's certificate chain and private key as a tonic identity.
    async fn load_identity(&self) -> Result<Identity> {
        let cert = fs::read(&self.config.server_cert_path).await?;
        let key = fs::read(&self.config.server_key_path).await?;
        Ok(Identity::from_pem(cert, key))
    }

    /// What the mesh listener serves: this node's certificate and key, and,
    /// when client certificates are required, the CA as the trust root the
    /// peers' certificates must chain to.
    pub async fn server_tls_config(&self) -> Result<ServerTlsConfig> {
        let mut tls = ServerTlsConfig::new().identity(self.load_identity().await?);
        if self.config.require_client_cert {
            tls = tls.client_ca_root(self.load_ca_certificate().await?);
        }
        Ok(tls)
    }

    /// What a dial to the peer named `server_name` trusts and presents: the
    /// CA as the trust root, this node's certificate as its identity.
    pub async fn client_tls_config(&self, server_name: &str) -> Result<ClientTlsConfig> {
        Ok(ClientTlsConfig::new()
            .domain_name(server_name)
            .ca_certificate(self.load_ca_certificate().await?)
            .identity(self.load_identity().await?))
    }

    /// Load certificates from file
    async fn load_certs(&self, path: &Path) -> Result<Vec<CertificateDer<'static>>> {
        let cert_data = fs::read(path).await?;
        let certs = certs(&mut cert_data.as_slice()).collect::<Result<Vec<_>, _>>()?;
        Ok(certs)
    }
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

    use tonic::transport::{ClientTlsConfig, Endpoint};

    use super::tls_server_name;

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
}
