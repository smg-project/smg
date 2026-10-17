//! Client-certificate authentication on the HTTPS listener, what it costs:
//! reading a verified leaf's identity (once per connection), and a full TLS
//! handshake on the loopback with and without a client certificate to verify
//! (the server's verification and the client's extra flight), each as one
//! connection that completes its handshake and closes.
//!
//! Run: `cargo bench -p smg --bench tls_client_auth`
#![expect(
    clippy::expect_used,
    reason = "a benchmark's setup fails loudly or not at all"
)]

#[path = "../tests/common/test_certs.rs"]
#[expect(
    unused_imports,
    reason = "the shared helper's own test module is compiled here without a test harness"
)]
mod test_certs;

use std::{hint::black_box, io::BufReader, net::TcpListener, sync::Arc};

use axum::{routing::get, Router};
use criterion::{criterion_group, criterion_main, Criterion};
use openssl::x509::X509;
use rustls::{pki_types::ServerName, ClientConfig, RootCertStore};
use rustls_pemfile::{certs, pkcs8_private_keys};
use smg::{
    config::{TlsClientAuth, TlsClientIdentity},
    server::client_identity::{
        identity_of, rustls_config_with_client_auth, ClientIdentityAcceptor,
    },
};
use test_certs::TestCertificates;
use tokio::{net::TcpStream, runtime::Runtime};
use tokio_rustls::TlsConnector;

const SPIFFE: &str = "spiffe://cluster.local/ns/default/sa/team-red";

fn certs_of(path: &std::path::Path) -> Vec<rustls::pki_types::CertificateDer<'static>> {
    certs(&mut BufReader::new(
        std::fs::File::open(path).expect("a file"),
    ))
    .filter_map(Result::ok)
    .collect()
}

/// A listener with `mode`, serving `/health`; its port.
fn listener(rt: &Runtime, certs: &TestCertificates, mode: TlsClientAuth) -> u16 {
    let read = |path: &std::path::Path| std::fs::read(path).expect("a file");
    let config = match mode {
        TlsClientAuth::Off => rt
            .block_on(axum_server::tls_rustls::RustlsConfig::from_pem(
                read(&certs.server_cert_path),
                read(&certs.server_key_path),
            ))
            .expect("a TLS configuration"),
        mode => rustls_config_with_client_auth(
            &read(&certs.server_cert_path),
            &read(&certs.server_key_path),
            &read(&certs.ca_cert_path),
            mode,
        )
        .expect("a TLS configuration with client authentication"),
    };
    let listener = TcpListener::bind("127.0.0.1:0").expect("a port");
    listener
        .set_nonblocking(true)
        .expect("a non-blocking listener");
    let port = listener.local_addr().expect("the address").port();
    let app = Router::new().route("/health", get(|| async { "ok" }));
    let field = mode.is_on().then_some(TlsClientIdentity::SanUri);
    rt.spawn(async move {
        axum_server::from_tcp_rustls(listener, config)
            .expect("a server")
            .map(|acceptor| ClientIdentityAcceptor::new(acceptor, field))
            .serve(app.into_make_service())
            .await
            .expect("the server runs");
    });
    port
}

/// A client trusting the CA, with the identity when given.
fn client(
    certs: &TestCertificates,
    identity: Option<&(std::path::PathBuf, std::path::PathBuf)>,
) -> TlsConnector {
    let mut roots = RootCertStore::empty();
    for cert in certs_of(&certs.ca_cert_path) {
        roots.add(cert).expect("a CA");
    }
    let builder = ClientConfig::builder().with_root_certificates(roots);
    let config = match identity {
        Some((cert, key)) => {
            let key = pkcs8_private_keys(&mut BufReader::new(
                std::fs::File::open(key).expect("a key"),
            ))
            .next()
            .expect("a key")
            .expect("a PKCS#8 key");
            builder
                .with_client_auth_cert(certs_of(cert), key.into())
                .expect("a client identity")
        }
        None => builder.with_no_client_auth(),
    };
    TlsConnector::from(Arc::new(config))
}

/// One connection: TCP connect, TLS handshake, close.
async fn handshake(connector: &TlsConnector, port: u16) {
    let tcp = TcpStream::connect(("127.0.0.1", port))
        .await
        .expect("connect");
    let tls = connector
        .connect(ServerName::try_from("localhost").expect("a name"), tcp)
        .await
        .expect("a handshake");
    drop(tls);
}

fn bench(c: &mut Criterion) {
    smg::tls::install_crypto_provider();
    let rt = Runtime::new().expect("a runtime");
    let certs = TestCertificates::generate().expect("certificates");
    let identity = certs
        .client_identity("team-red", SPIFFE)
        .expect("an identity");
    let leaf = X509::from_pem(&std::fs::read(&identity.0).expect("the certificate"))
        .expect("a certificate")
        .to_der()
        .expect("DER");

    let mut group = c.benchmark_group("tls_client_auth");
    group.bench_function("identity_of_leaf_san_uri", |b| {
        b.iter(|| black_box(identity_of(black_box(&leaf), TlsClientIdentity::SanUri)));
    });
    group.bench_function("identity_of_leaf_cn", |b| {
        b.iter(|| black_box(identity_of(black_box(&leaf), TlsClientIdentity::Cn)));
    });

    let plain = listener(&rt, &certs, TlsClientAuth::Off);
    let required = listener(&rt, &certs, TlsClientAuth::Required);
    let bare = client(&certs, None);
    let identified = client(&certs, Some(&identity));
    // Warm both listeners up.
    rt.block_on(handshake(&bare, plain));
    rt.block_on(handshake(&identified, required));

    group.sample_size(50);
    group.bench_function("handshake_no_client_auth", |b| {
        b.iter(|| rt.block_on(handshake(&bare, plain)));
    });
    group.bench_function("handshake_required_with_certificate", |b| {
        b.iter(|| rt.block_on(handshake(&identified, required)));
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
