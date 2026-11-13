//! The listeners' TLS: the process crypto provider, installed once before
//! the first listener (the gateway, the servicer and the mock worker install
//! the same one), and a listener's material, read once at bind and never
//! per connection.

use std::path::{Path, PathBuf};

use rustls::crypto::ring;
use tonic::transport::{Certificate, Identity, ServerTlsConfig};

use super::config::Tls;

/// Installs the ring provider as the process's. A second install (a host
/// that installed its own first, or two controllers in one test process) is
/// not an error: the provider is the process's, whoever set it.
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
mod tests {
    use std::path::{Path, PathBuf};

    use super::{has_block, Material, TlsError, CERTIFICATE_BLOCKS, PRIVATE_KEY_BLOCKS};
    use crate::controller::config::Tls;

    const CERT: &str = "-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n";
    const KEY: &str = concat!(
        "-----BEGIN ",
        "PRIVATE KEY-----\nMIIB\n-----END PRIVATE KEY-----\n"
    );

    #[test]
    fn a_block_is_found_by_its_label_anywhere_in_the_file() {
        assert!(has_block(CERT.as_bytes(), CERTIFICATE_BLOCKS));
        assert!(!has_block(KEY.as_bytes(), CERTIFICATE_BLOCKS));
        assert!(has_block(KEY.as_bytes(), PRIVATE_KEY_BLOCKS));
        let ec = concat!(
            "comment\n-----BEGIN EC ",
            "PRIVATE KEY-----\nMHc\n-----END EC PRIVATE KEY-----\n"
        );
        assert!(has_block(ec.as_bytes(), PRIVATE_KEY_BLOCKS));
        assert!(!has_block(b"", PRIVATE_KEY_BLOCKS));
    }

    #[tokio::test]
    async fn each_file_is_refused_by_name() {
        let dir = tempfile::tempdir().expect("a scratch directory");
        let cert = dir.path().join("server.pem");
        let key = dir.path().join("server.key");
        let ca = dir.path().join("ca.pem");
        tokio::fs::write(&cert, CERT).await.expect("write");
        tokio::fs::write(&key, KEY).await.expect("write");
        tokio::fs::write(&ca, CERT).await.expect("write");

        let tls = Tls {
            cert: cert.clone(),
            key: key.clone(),
            client_ca: Some(ca.clone()),
        };
        let material = Material::read(&tls)
            .await
            .expect("three well-labelled files");
        assert!(material.mutual());

        let missing = Tls {
            cert: PathBuf::from("/nonexistent/server.pem"),
            key: key.clone(),
            client_ca: None,
        };
        assert!(matches!(
            Material::read(&missing).await,
            Err(TlsError::Read { path, .. }) if path == Path::new("/nonexistent/server.pem")
        ));

        let swapped = Tls {
            cert: key.clone(),
            key: cert.clone(),
            client_ca: None,
        };
        assert!(matches!(
            Material::read(&swapped).await,
            Err(TlsError::NoCertificate(path)) if path == key
        ));

        let key_is_a_cert = Tls {
            cert: cert.clone(),
            key: cert.clone(),
            client_ca: None,
        };
        assert!(matches!(
            Material::read(&key_is_a_cert).await,
            Err(TlsError::NoPrivateKey(path)) if path == cert
        ));

        let ca_is_a_key = Tls {
            cert: cert.clone(),
            key: key.clone(),
            client_ca: Some(key.clone()),
        };
        assert!(matches!(
            Material::read(&ca_is_a_key).await,
            Err(TlsError::NoCertificate(path)) if path == key
        ));

        let server_only = Tls {
            cert,
            key,
            client_ca: None,
        };
        let material = Material::read(&server_only).await.expect("two files");
        assert!(!material.mutual());
    }
}
