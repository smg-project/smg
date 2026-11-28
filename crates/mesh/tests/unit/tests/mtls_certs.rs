//! Certificates for the mTLS tests: a CA and a node certificate it signed,
//! with IP SANs for both loopback addresses and a DNS SAN for `localhost`,
//! written as PEM files because the mTLS configuration takes paths.

use std::path::PathBuf;

use openssl::{
    asn1::{Asn1Integer, Asn1Time},
    bn::{BigNum, MsbOption},
    ec::{EcGroup, EcKey},
    hash::MessageDigest,
    nid::Nid,
    pkey::{PKey, Private},
    x509::{
        extension::{BasicConstraints, ExtendedKeyUsage, KeyUsage, SubjectAlternativeName},
        X509Builder, X509Name, X509NameBuilder, X509,
    },
};

/// PEM files of a test CA and of a node certificate it signed, in a private
/// temporary directory of their own (created exclusively, removed on drop).
pub(crate) struct MtlsTestCerts {
    _dir: tempfile::TempDir,
    pub(crate) ca_cert_path: PathBuf,
    pub(crate) node_cert_path: PathBuf,
    pub(crate) node_key_path: PathBuf,
}

impl MtlsTestCerts {
    /// A fresh CA and a node certificate with the SANs `IP:::1`,
    /// `IP:127.0.0.1` and `DNS:localhost`.
    pub(crate) fn generate() -> Self {
        let dir = tempfile::Builder::new()
            .prefix("smg-mesh-mtls-")
            .tempdir()
            .expect("create the certificate directory");

        let ca_key = p256_key();
        let ca_cert = ca_certificate(&ca_key);
        let node_key = p256_key();
        let node_cert = node_certificate(&node_key, &ca_cert, &ca_key);

        let ca_cert_path = dir.path().join("ca.pem");
        let node_cert_path = dir.path().join("node.pem");
        let node_key_path = dir.path().join("node-key.pem");
        std::fs::write(&ca_cert_path, ca_cert.to_pem().expect("CA PEM")).expect("write the CA");
        std::fs::write(&node_cert_path, node_cert.to_pem().expect("node PEM"))
            .expect("write the node certificate");
        std::fs::write(
            &node_key_path,
            node_key
                .private_key_to_pem_pkcs8()
                .expect("node key as PKCS#8 PEM"),
        )
        .expect("write the node key");

        Self {
            _dir: dir,
            ca_cert_path,
            node_cert_path,
            node_key_path,
        }
    }
}

fn p256_key() -> PKey<Private> {
    let group = EcGroup::from_curve_name(Nid::X9_62_PRIME256V1).expect("the P-256 group");
    let key = EcKey::generate(&group).expect("generate a P-256 key");
    PKey::from_ec_key(key).expect("wrap the EC key")
}

fn common_name(cn: &str) -> X509Name {
    let mut name = X509NameBuilder::new().expect("name builder");
    name.append_entry_by_text("CN", cn).expect("append the CN");
    name.build()
}

fn random_serial() -> Asn1Integer {
    let mut serial = BigNum::new().expect("serial");
    serial
        .rand(128, MsbOption::MAYBE_ZERO, false)
        .expect("random serial");
    serial.to_asn1_integer().expect("serial as ASN.1 integer")
}

fn ca_certificate(key: &PKey<Private>) -> X509 {
    let name = common_name("mesh test CA");
    let mut cert = X509Builder::new().expect("certificate builder");
    cert.set_version(2).expect("X.509 v3");
    cert.set_serial_number(&random_serial()).expect("serial");
    cert.set_subject_name(&name).expect("subject");
    cert.set_issuer_name(&name).expect("issuer");
    cert.set_pubkey(key).expect("public key");
    cert.set_not_before(&Asn1Time::days_from_now(0).expect("now"))
        .expect("not before");
    cert.set_not_after(&Asn1Time::days_from_now(1).expect("tomorrow"))
        .expect("not after");
    cert.append_extension(
        BasicConstraints::new()
            .critical()
            .ca()
            .build()
            .expect("basic constraints"),
    )
    .expect("append basic constraints");
    cert.append_extension(
        KeyUsage::new()
            .critical()
            .key_cert_sign()
            .crl_sign()
            .build()
            .expect("key usage"),
    )
    .expect("append key usage");
    cert.sign(key, MessageDigest::sha256())
        .expect("sign the CA");
    cert.build()
}

fn node_certificate(key: &PKey<Private>, ca_cert: &X509, ca_key: &PKey<Private>) -> X509 {
    let mut cert = X509Builder::new().expect("certificate builder");
    cert.set_version(2).expect("X.509 v3");
    cert.set_serial_number(&random_serial()).expect("serial");
    cert.set_subject_name(&common_name("mesh test node"))
        .expect("subject");
    cert.set_issuer_name(ca_cert.subject_name())
        .expect("issuer");
    cert.set_pubkey(key).expect("public key");
    cert.set_not_before(&Asn1Time::days_from_now(0).expect("now"))
        .expect("not before");
    cert.set_not_after(&Asn1Time::days_from_now(1).expect("tomorrow"))
        .expect("not after");
    cert.append_extension(BasicConstraints::new().build().expect("basic constraints"))
        .expect("append basic constraints");
    cert.append_extension(
        KeyUsage::new()
            .critical()
            .digital_signature()
            .key_encipherment()
            .build()
            .expect("key usage"),
    )
    .expect("append key usage");
    cert.append_extension(
        ExtendedKeyUsage::new()
            .server_auth()
            .client_auth()
            .build()
            .expect("extended key usage"),
    )
    .expect("append extended key usage");
    let san = SubjectAlternativeName::new()
        .ip("::1")
        .ip("127.0.0.1")
        .dns("localhost")
        .build(&cert.x509v3_context(Some(ca_cert), None))
        .expect("subject alternative names");
    cert.append_extension(san).expect("append the SANs");
    cert.sign(ca_key, MessageDigest::sha256())
        .expect("sign the node certificate");
    cert.build()
}
