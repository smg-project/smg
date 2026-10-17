//! Client-certificate authentication on the HTTPS listener, over real
//! handshakes against the listener's acceptor chain and the serving routes'
//! authentication: a verified certificate's identity is the caller's tenant,
//! `required` turns away a client without one, `optional` lets the keys
//! authenticate such a client, and a certificate from another CA or without
//! the identity field is refused.

use std::{
    io::BufReader,
    net::{SocketAddr, TcpListener},
    path::Path,
    time::Duration,
};

use axum::{
    body::Body, extract::Request, http::header, middleware::from_fn_with_state,
    response::IntoResponse, routing::get, Router,
};
use axum_server::Handle;
use rustls::{ClientConfig, RootCertStore};
use rustls_pemfile::{certs, pkcs8_private_keys};
use smg::{
    config::{TlsClientAuth, TlsClientIdentity},
    middleware::{auth_middleware, AuthConfig, DataPlaneCaller},
    server::client_identity::{rustls_config_with_client_auth, ClientIdentityAcceptor},
};

use crate::common::test_certs::TestCertificates;

const SPIFFE: &str = "spiffe://cluster.local/ns/default/sa/team-red";

/// The listener: the acceptor chain of the HTTPS listener under `mode` with
/// `field` as the identity, over the serving auth (the key `k` configured)
/// and a route that answers the tenant key it resolved. Shut down with the
/// handle.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test helper: failures panic; the server lives as long as the test"
)]
fn listener(
    certs: &TestCertificates,
    mode: TlsClientAuth,
    field: TlsClientIdentity,
) -> (String, Handle<SocketAddr>) {
    let config = rustls_config_with_client_auth(
        &std::fs::read(&certs.server_cert_path).unwrap(),
        &std::fs::read(&certs.server_key_path).unwrap(),
        &std::fs::read(&certs.ca_cert_path).unwrap(),
        mode,
    )
    .expect("a TLS configuration with client authentication");
    let auth = AuthConfig::new(Some("k".to_string())).with_client_certificates(mode);
    let app = Router::new()
        .route(
            "/",
            get(|request: Request<Body>| async move {
                request
                    .extensions()
                    .get::<DataPlaneCaller>()
                    .map(|caller| caller.tenant_key().to_string())
                    .unwrap_or_else(|| "missing".to_string())
                    .into_response()
            }),
        )
        .route_layer(from_fn_with_state(auth, auth_middleware));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!(
        "https://localhost:{}/",
        listener.local_addr().unwrap().port()
    );
    let handle = Handle::new();
    let server_handle = handle.clone();
    tokio::spawn(async move {
        axum_server::from_tcp_rustls(listener, config)
            .unwrap()
            .map(|acceptor| ClientIdentityAcceptor::new(acceptor, Some(field)))
            .handle(server_handle)
            .serve(app.into_make_service())
            .await
            .unwrap();
    });
    (url, handle)
}

/// A client trusting the test CA, presenting the certificate at `identity`
/// when given.
#[expect(clippy::unwrap_used, reason = "test helper: failures panic")]
fn client(certs: &TestCertificates, identity: Option<(&Path, &Path)>) -> reqwest::Client {
    let mut roots = RootCertStore::empty();
    let ca = std::fs::File::open(&certs.ca_cert_path).unwrap();
    for cert in certs_of(&mut BufReader::new(ca)) {
        roots.add(cert).unwrap();
    }
    let builder = ClientConfig::builder().with_root_certificates(roots);
    let config = match identity {
        Some((cert, key)) => {
            let chain = certs_of(&mut BufReader::new(std::fs::File::open(cert).unwrap()));
            let key = pkcs8_private_keys(&mut BufReader::new(std::fs::File::open(key).unwrap()))
                .next()
                .unwrap()
                .unwrap();
            builder.with_client_auth_cert(chain, key.into()).unwrap()
        }
        None => builder.with_no_client_auth(),
    };
    smg::http_client::client_builder()
        .use_preconfigured_tls(config)
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap()
}

fn certs_of(reader: &mut dyn std::io::BufRead) -> Vec<rustls::pki_types::CertificateDer<'static>> {
    certs(reader).filter_map(Result::ok).collect()
}

/// `GET /` with `authorization`: the status and body, or the transport error
/// (a failed handshake).
async fn https_get(
    client: &reqwest::Client,
    url: &str,
    authorization: Option<&str>,
) -> Result<(u16, String), String> {
    let mut request = client.get(url);
    if let Some(authorization) = authorization {
        request = request.header(header::AUTHORIZATION, authorization);
    }
    let response = request.send().await.map_err(|e| e.to_string())?;
    let status = response.status().as_u16();
    let body = response.text().await.map_err(|e| e.to_string())?;
    Ok((status, body))
}

#[tokio::test]
async fn a_required_client_certificate_is_the_tenant_and_its_absence_is_refused() {
    smg::tls::install_crypto_provider();
    let certs = TestCertificates::generate().unwrap();
    let (cert, key) = certs.client_identity("team-red", SPIFFE).unwrap();
    let other_ca = TestCertificates::generate().unwrap();
    let (foreign_cert, foreign_key) = other_ca.client_identity("intruder", SPIFFE).unwrap();
    let (url, handle) = listener(&certs, TlsClientAuth::Required, TlsClientIdentity::SanUri);

    let identified = client(&certs, Some((&cert, &key)));
    assert_eq!(
        https_get(&identified, &url, None).await.unwrap(),
        (200, format!("auth:{SPIFFE}")),
        "the URI SAN is the tenant, no bearer needed"
    );
    assert_eq!(
        https_get(&identified, &url, Some("Bearer not-a-key"))
            .await
            .unwrap()
            .0,
        200,
        "a bearer on a certificate connection is not consulted"
    );

    let bare = client(&certs, None);
    let refused = https_get(&bare, &url, Some("Bearer k")).await;
    assert!(
        refused.is_err(),
        "no certificate: no connection, key or not: {refused:?}"
    );

    let foreign = client(&certs, Some((&foreign_cert, &foreign_key)));
    let refused = https_get(&foreign, &url, None).await;
    assert!(refused.is_err(), "another CA's certificate: {refused:?}");

    // The CA's plain client certificate carries no SAN at all: verified,
    // but no identity in the configured field.
    let unidentified = client(
        &certs,
        Some((&certs.client_cert_path, &certs.client_key_path)),
    );
    let (status, body) = https_get(&unidentified, &url, None).await.unwrap();
    assert_eq!(status, 403, "{body}");
    assert!(body.contains("client_certificate_unidentified"), "{body}");

    handle.shutdown();
}

#[tokio::test]
async fn an_optional_client_certificate_identifies_and_the_keys_cover_the_rest() {
    smg::tls::install_crypto_provider();
    let certs = TestCertificates::generate().unwrap();
    let (cert, key) = certs.client_identity("team-red", SPIFFE).unwrap();
    let (url, handle) = listener(&certs, TlsClientAuth::Optional, TlsClientIdentity::SanDns);

    let identified = client(&certs, Some((&cert, &key)));
    assert_eq!(
        https_get(&identified, &url, None).await.unwrap(),
        (200, "auth:team-red".to_string()),
        "the DNS SAN is the tenant"
    );

    let bare = client(&certs, None);
    let (status, _) = https_get(&bare, &url, None).await.unwrap();
    assert_eq!(status, 401, "no certificate and no key");
    let (status, body) = https_get(&bare, &url, Some("Bearer k")).await.unwrap();
    assert_eq!(status, 200, "{body}");
    assert!(body.starts_with("auth:"), "the key's tenant: {body}");

    handle.shutdown();
}

#[tokio::test]
async fn the_common_name_can_be_the_identity() {
    smg::tls::install_crypto_provider();
    let certs = TestCertificates::generate().unwrap();
    let (cert, key) = certs.client_identity("team-blue", SPIFFE).unwrap();
    let (url, handle) = listener(&certs, TlsClientAuth::Required, TlsClientIdentity::Cn);

    let identified = client(&certs, Some((&cert, &key)));
    assert_eq!(
        https_get(&identified, &url, None).await.unwrap(),
        (200, "auth:team-blue".to_string())
    );

    handle.shutdown();
}
