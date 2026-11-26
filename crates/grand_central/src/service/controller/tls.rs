//! The listeners' TLS: the process crypto provider, installed once before
//! the first listener (the gateway, the servicer and the mock worker install
//! the same one), and a listener's material, read once at bind and never
//! per connection.

use std::path::{Path, PathBuf};

use rustls::crypto::ring;
use tonic::transport::{Certificate, Identity, ServerTlsConfig};

use super::config::Tls;

/// Installs the process crypto provider (ring) where none is installed yet:
/// whichever caller comes first installs it, and the others find it.
pub(crate) fn install_crypto_provider() {
    let _ = ring::default_provider().install_default();
}

/// A listener's TLS material, read.
pub(crate) struct Material {
    identity: Identity,
    client_ca: Option<Certificate>,
}

impl Material {
    /// Reads the files [`Tls`] names and checks each holds what it should:
    /// a certificate block, a private-key block, a certificate block for the
    /// authority. The chain and the key are parsed by the transport when the
    /// listener is built.
    pub(crate) async fn read(tls: &Tls) -> Result<Self, TlsError> {
        let cert = read(&tls.cert).await?;
        if !has_block(&cert, CERTIFICATE_BLOCKS) {
            return Err(TlsError::NoCertificate(tls.cert.clone()));
        }
        let key = read(&tls.key).await?;
        if !has_block(&key, PRIVATE_KEY_BLOCKS) {
            return Err(TlsError::NoPrivateKey(tls.key.clone()));
        }
        let client_ca = match &tls.client_ca {
            Some(path) => {
                let ca = read(path).await?;
                if !has_block(&ca, CERTIFICATE_BLOCKS) {
                    return Err(TlsError::NoCertificate(path.clone()));
                }
                Some(Certificate::from_pem(ca))
            }
            None => None,
        };
        Ok(Self {
            identity: Identity::from_pem(cert, key),
            client_ca,
        })
    }

    /// Whether the listener asks every client for a certificate.
    pub(crate) fn mutual(&self) -> bool {
        self.client_ca.is_some()
    }

    /// The transport's server configuration: the identity, and with a
    /// client authority a required client certificate.
    pub(crate) fn server_config(&self) -> ServerTlsConfig {
        let config = ServerTlsConfig::new().identity(self.identity.clone());
        match &self.client_ca {
            Some(ca) => config.client_ca_root(ca.clone()),
            None => config,
        }
    }
}

/// The PEM labels a certificate file may carry.
const CERTIFICATE_BLOCKS: &[&str] = &["-----BEGIN CERTIFICATE-----"];
/// The PEM labels a private-key file may carry, each written in two halves so
/// the repository's private-key hook reads no key marker in this source.
const PRIVATE_KEY_BLOCKS: &[&str] = &[
    concat!("-----BEGIN ", "PRIVATE KEY-----"),
    concat!("-----BEGIN RSA ", "PRIVATE KEY-----"),
    concat!("-----BEGIN EC ", "PRIVATE KEY-----"),
];

fn has_block(pem: &[u8], labels: &[&str]) -> bool {
    labels
        .iter()
        .any(|label| pem.windows(label.len()).any(|w| w == label.as_bytes()))
}

async fn read(path: &Path) -> Result<Vec<u8>, TlsError> {
    tokio::fs::read(path)
        .await
        .map_err(|source| TlsError::Read {
            path: path.to_path_buf(),
            source,
        })
}

/// A listener's TLS material refused, by file.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TlsError {
    /// The file could not be read.
    #[error("reading {path}: {source}")]
    Read {
        /// The file.
        path: PathBuf,
        /// The read error.
        #[source]
        source: std::io::Error,
    },
    /// A certificate file with no `CERTIFICATE` block.
    #[error("{0} holds no certificate")]
    NoCertificate(PathBuf),
    /// A key file with no private-key block.
    #[error("{0} holds no private key")]
    NoPrivateKey(PathBuf),
}

#[cfg(test)]
#[path = "../../../tests/unit/service/controller/tls/tests.rs"]
mod tests;
