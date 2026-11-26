use std::path::{Path, PathBuf};

use super::{has_block, Material, TlsError, CERTIFICATE_BLOCKS, PRIVATE_KEY_BLOCKS};
use crate::service::controller::config::Tls;

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
