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
#[path = "../tests/unit/mtls/tests.rs"]
mod tests;
