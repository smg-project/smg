//! Client-certificate identity on the HTTPS listener (`--tls-client-auth`).
//!
//! With client authentication on, the listener's TLS configuration
//! ([`rustls_config_with_client_auth`]) asks every client for a certificate
//! and verifies the one presented against the configured CA bundle during
//! the handshake: under `required` a client without one fails the
//! handshake, under `optional` it is served without a certificate identity.
//! [`ClientIdentityAcceptor`] runs after the handshake, reads the verified
//! leaf once per connection and hands every request of the connection a
//! [`ClientCertificate`] extension: the identity the configured field of
//! the leaf names (the first URI subject alternative name, a SPIFFE id; the
//! first DNS name; or the subject's common name), or the fact that the leaf
//! has none. The serving routes' authentication (`middleware::auth`) turns
//! the identity into the caller's tenant. Nothing here trusts the
//! certificate on its own: only a leaf rustls verified reaches the acceptor.

use std::{future::Future, io, pin::Pin, sync::Arc};

use axum_server::{accept::Accept, tls_rustls::RustlsConfig};
use http::Request;
use openssl::{nid::Nid, x509::X509};
use rustls::{
    pki_types::CertificateDer, server::WebPkiClientVerifier, RootCertStore, ServerConfig,
};
use tokio_rustls::server::TlsStream;
use tower::Service;
use tracing::{debug, warn};

use crate::{
    config::{TlsClientAuth, TlsClientIdentity},
    observability::metrics::Metrics,
    tenant::MAX_TENANT_ID_LEN,
};

/// What the connection's verified client certificate says about the caller.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClientCertificate {
    /// The identity the configured field of the leaf names.
    Identified(Arc<str>),
    /// A verified leaf without the configured field (or with an unusable
    /// value in it).
    Unidentified,
}

/// The identity a leaf certificate (DER) names in `field`: the first URI or
/// DNS subject alternative name of that kind, or the subject's common name.
/// `None` when the leaf has none, carries an empty, padded, over-long or
/// control-character one (the identity becomes a tenant key, bounded like a
/// tenant header), or cannot be parsed.
pub fn identity_of(leaf: &[u8], field: TlsClientIdentity) -> Option<String> {
    let certificate = X509::from_der(leaf).ok()?;
    let identity = match field {
        TlsClientIdentity::SanUri => certificate
            .subject_alt_names()?
            .iter()
            .find_map(|name| name.uri().map(str::to_owned))?,
        TlsClientIdentity::SanDns => certificate
            .subject_alt_names()?
            .iter()
            .find_map(|name| name.dnsname().map(str::to_owned))?,
        TlsClientIdentity::Cn => {
            let common_name = certificate
                .subject_name()
                .entries_by_nid(Nid::COMMONNAME)
                .next()?;
            // Through OpenSSL's conversion, so every ASN.1 string type (a
            // BMPString common name is UTF-16) arrives as UTF-8; the raw
            // bytes would not.
            common_name.data().to_string().ok()?
        }
    };
    // The identity is embedded verbatim in the tenant key (`auth:<identity>`):
    // a padded value or one with control characters is no identity, as a
    // padded tenant id is no tenant key.
    let usable = !identity.is_empty()
        && identity.len() <= MAX_TENANT_ID_LEN
        && identity.trim() == identity
        && !identity.chars().any(char::is_control);
    usable.then_some(identity)
}

/// Streams that know the peer's certificate chain: the TLS stream, once its
/// handshake is complete.
pub trait PeerCertificates {
    /// The leaf of the verified chain the peer presented, DER.
    fn peer_leaf(&self) -> Option<&CertificateDer<'static>>;
}

impl<IO> PeerCertificates for TlsStream<IO> {
    fn peer_leaf(&self) -> Option<&CertificateDer<'static>> {
        self.get_ref()
            .1
            .peer_certificates()
            .and_then(|chain| chain.first())
    }
}

/// An acceptor that reads the identity of the connection's client
/// certificate once the inner acceptor has completed the TLS handshake, and
/// hands it to every request of the connection through [`IdentityService`].
/// With `field` unset (client authentication off) it reads nothing and
/// passes the connection through.
#[derive(Clone, Debug)]
pub struct ClientIdentityAcceptor<A> {
    inner: A,
    field: Option<TlsClientIdentity>,
}

impl<A> ClientIdentityAcceptor<A> {
    pub fn new(inner: A, field: Option<TlsClientIdentity>) -> Self {
        Self { inner, field }
    }
}

impl<A, I, S> Accept<I, S> for ClientIdentityAcceptor<A>
where
    A: Accept<I, S>,
    A::Future: Send + 'static,
    A::Stream: PeerCertificates + Send + 'static,
    A::Service: Send + 'static,
{
    type Stream = A::Stream;
    type Service = IdentityService<A::Service>;
    type Future = Pin<Box<dyn Future<Output = io::Result<(Self::Stream, Self::Service)>> + Send>>;

    fn accept(&self, stream: I, service: S) -> Self::Future {
        let inner = self.inner.accept(stream, service);
        let field = self.field;
        Box::pin(async move {
            let Some(field) = field else {
                let (stream, inner) = inner.await?;
                return Ok((
                    stream,
                    IdentityService {
                        inner,
                        certificate: None,
                    },
                ));
            };
            let (stream, inner) = match inner.await {
                Ok(accepted) => accepted,
                Err(error) => {
                    Metrics::record_tls_client_auth("handshake_failed");
                    return Err(error);
                }
            };
            let certificate = stream.peer_leaf().map(|leaf| match identity_of(leaf, field) {
                Some(identity) => {
                    Metrics::record_tls_client_auth("identified");
                    ClientCertificate::Identified(Arc::from(identity))
                }
                None => {
                    debug!(
                        field = field.as_str(),
                        "a verified client certificate names no usable identity in the configured field"
                    );
                    Metrics::record_tls_client_auth("unidentified");
                    ClientCertificate::Unidentified
                }
            });
            if certificate.is_none() {
                Metrics::record_tls_client_auth("none");
            }
            Ok((stream, IdentityService { inner, certificate }))
        })
    }
}

/// The connection's service, handing each request the connection's
/// [`ClientCertificate`] when it presented one.
#[derive(Clone, Debug)]
pub struct IdentityService<S> {
    inner: S,
    certificate: Option<ClientCertificate>,
}

impl<S, B> Service<Request<B>> for IdentityService<S>
where
    S: Service<Request<B>>,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = S::Future;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut request: Request<B>) -> Self::Future {
        if let Some(certificate) = &self.certificate {
            request.extensions_mut().insert(certificate.clone());
        }
        self.inner.call(request)
    }
}

/// The listener's TLS configuration with client authentication: the server
/// certificate chain and key (PEM), the CA bundle (PEM) client certificates
/// must chain to, and the mode (`Optional` admits a connection without a
/// certificate, `Required` fails its handshake). ALPN as without client
/// authentication (HTTP/2 and HTTP/1.1). An error names what is wrong with
/// the material, never its content.
pub fn rustls_config_with_client_auth(
    cert_pem: &[u8],
    key_pem: &[u8],
    client_ca_pem: &[u8],
    mode: TlsClientAuth,
) -> Result<RustlsConfig, String> {
    crate::tls::install_crypto_provider();
    let certs = rustls_pemfile::certs(&mut &*cert_pem)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("the server certificate is not a PEM certificate chain: {e}"))?;
    let key = rustls_pemfile::private_key(&mut &*key_pem)
        .map_err(|e| format!("the server key is not a PEM private key: {e}"))?
        .ok_or("the server key file holds no private key")?;
    let client_cas = rustls_pemfile::certs(&mut &*client_ca_pem)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("the client CA bundle is not a PEM certificate bundle: {e}"))?;
    let mut roots = RootCertStore::empty();
    let (added, ignored) = roots.add_parsable_certificates(client_cas);
    if added == 0 {
        return Err("the client CA bundle holds no usable CA certificate".to_string());
    }
    if ignored > 0 {
        warn!(
            added,
            ignored,
            "some certificates of the client CA bundle could not be parsed as CAs and are ignored"
        );
    }
    let verifier = WebPkiClientVerifier::builder(Arc::new(roots));
    let verifier = match mode {
        TlsClientAuth::Required => verifier,
        TlsClientAuth::Optional => verifier.allow_unauthenticated(),
        TlsClientAuth::Off => {
            return Err("client authentication is off: no verifier to build".to_string())
        }
    }
    .build()
    .map_err(|e| format!("the client certificate verifier could not be built: {e}"))?;
    let mut config = ServerConfig::builder()
        .with_client_cert_verifier(verifier)
        .with_single_cert(certs, key)
        .map_err(|e| format!("the server certificate and key do not form a TLS identity: {e}"))?;
    config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    Ok(RustlsConfig::from_config(Arc::new(config)))
}

#[cfg(test)]
mod tests {
    use std::{net::TcpListener, time::Duration};

    use axum::{body::Body, routing::get, Router};
    use http::Response;
    use openssl::{
        asn1::{Asn1Time, Asn1Type},
        bn::{BigNum, MsbOption},
        hash::MessageDigest,
        pkey::{PKey, Private},
        rsa::Rsa,
        x509::{
            extension::{BasicConstraints, ExtendedKeyUsage, KeyUsage, SubjectAlternativeName},
            X509Builder, X509NameBuilder, X509,
        },
    };
    use rustls::{pki_types::ServerName, ClientConfig};
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpStream,
    };
    use tokio_rustls::TlsConnector;

    use super::*;

    const SPIFFE: &str = "spiffe://cluster.local/ns/default/sa/team-red";

    /// A CA and the leaves it signs, kept as PEM the way the gateway reads them.
    struct Pki {
        ca_cert: X509,
        ca_key: PKey<Private>,
        ca_pem: Vec<u8>,
    }

    fn key() -> PKey<Private> {
        PKey::from_rsa(Rsa::generate(2048).unwrap()).unwrap()
    }

    fn serial() -> openssl::asn1::Asn1Integer {
        let mut bn = BigNum::new().unwrap();
        bn.rand(128, MsbOption::MAYBE_ZERO, false).unwrap();
        bn.to_asn1_integer().unwrap()
    }

    fn name(cn: &str) -> openssl::x509::X509Name {
        name_with_type(cn, Asn1Type::UTF8STRING)
    }

    /// A subject whose common name is encoded as the ASN.1 string type `ty`.
    fn name_with_type(cn: &str, ty: Asn1Type) -> openssl::x509::X509Name {
        let mut builder = X509NameBuilder::new().unwrap();
        builder.append_entry_by_text("O", "Test").unwrap();
        builder
            .append_entry_by_text_with_type("CN", cn, ty)
            .unwrap();
        builder.build()
    }

    impl Pki {
        fn new() -> Self {
            let ca_key = key();
            let mut builder = X509Builder::new().unwrap();
            builder.set_version(2).unwrap();
            builder.set_serial_number(&serial()).unwrap();
            let name = name("Test CA");
            builder.set_subject_name(&name).unwrap();
            builder.set_issuer_name(&name).unwrap();
            builder.set_pubkey(&ca_key).unwrap();
            builder
                .set_not_before(&Asn1Time::days_from_now(0).unwrap())
                .unwrap();
            builder
                .set_not_after(&Asn1Time::days_from_now(1).unwrap())
                .unwrap();
            builder
                .append_extension(BasicConstraints::new().critical().ca().build().unwrap())
                .unwrap();
            builder
                .append_extension(
                    KeyUsage::new()
                        .critical()
                        .key_cert_sign()
                        .crl_sign()
                        .build()
                        .unwrap(),
                )
                .unwrap();
            builder.sign(&ca_key, MessageDigest::sha256()).unwrap();
            let ca_cert = builder.build();
            let ca_pem = ca_cert.to_pem().unwrap();
            Self {
                ca_cert,
                ca_key,
                ca_pem,
            }
        }

        /// A leaf signed by the CA with `cn` and the given SANs (URIs and
        /// DNS names), good as a server and as a client; its PEM and key.
        fn leaf(&self, cn: &str, uris: &[&str], dns: &[&str]) -> (Vec<u8>, Vec<u8>) {
            self.leaf_named(&name(cn), uris, dns)
        }

        /// A leaf with `subject` as its subject name.
        fn leaf_named(
            &self,
            subject: &openssl::x509::X509Name,
            uris: &[&str],
            dns: &[&str],
        ) -> (Vec<u8>, Vec<u8>) {
            let key = key();
            let mut builder = X509Builder::new().unwrap();
            builder.set_version(2).unwrap();
            builder.set_serial_number(&serial()).unwrap();
            builder.set_subject_name(subject).unwrap();
            builder
                .set_issuer_name(self.ca_cert.subject_name())
                .unwrap();
            builder.set_pubkey(&key).unwrap();
            builder
                .set_not_before(&Asn1Time::days_from_now(0).unwrap())
                .unwrap();
            builder
                .set_not_after(&Asn1Time::days_from_now(1).unwrap())
                .unwrap();
            builder
                .append_extension(BasicConstraints::new().build().unwrap())
                .unwrap();
            builder
                .append_extension(
                    KeyUsage::new()
                        .critical()
                        .digital_signature()
                        .key_encipherment()
                        .build()
                        .unwrap(),
                )
                .unwrap();
            builder
                .append_extension(
                    ExtendedKeyUsage::new()
                        .server_auth()
                        .client_auth()
                        .build()
                        .unwrap(),
                )
                .unwrap();
            if !uris.is_empty() || !dns.is_empty() {
                let mut san = SubjectAlternativeName::new();
                for uri in uris {
                    san.uri(uri);
                }
                for name in dns {
                    san.dns(name);
                }
                let san = san
                    .build(&builder.x509v3_context(Some(&self.ca_cert), None))
                    .unwrap();
                builder.append_extension(san).unwrap();
            }
            builder.sign(&self.ca_key, MessageDigest::sha256()).unwrap();
            let cert = builder.build();
            (
                cert.to_pem().unwrap(),
                key.private_key_to_pem_pkcs8().unwrap(),
            )
        }
    }

    fn der_of(pem: &[u8]) -> Vec<u8> {
        X509::from_pem(pem).unwrap().to_der().unwrap()
    }

    #[test]
    fn the_identity_is_the_first_name_of_the_configured_kind() {
        let pki = Pki::new();
        let (leaf, _) = pki.leaf(
            "team-red",
            &[SPIFFE, "spiffe://cluster.local/ns/default/sa/second"],
            &["team-red.example", "alt.example"],
        );
        let leaf = der_of(&leaf);

        assert_eq!(
            identity_of(&leaf, TlsClientIdentity::SanUri).as_deref(),
            Some(SPIFFE)
        );
        assert_eq!(
            identity_of(&leaf, TlsClientIdentity::SanDns).as_deref(),
            Some("team-red.example")
        );
        assert_eq!(
            identity_of(&leaf, TlsClientIdentity::Cn).as_deref(),
            Some("team-red")
        );
    }

    /// A leaf without the configured kind of name has no identity; a DNS
    /// name does not stand in for a URI, nor the common name for either.
    #[test]
    fn a_leaf_without_the_configured_name_has_no_identity() {
        let pki = Pki::new();
        let (dns_only, _) = pki.leaf("team-red", &[], &["team-red.example"]);
        let dns_only = der_of(&dns_only);
        assert_eq!(identity_of(&dns_only, TlsClientIdentity::SanUri), None);
        assert_eq!(
            identity_of(&dns_only, TlsClientIdentity::SanDns).as_deref(),
            Some("team-red.example")
        );

        let (no_san, _) = pki.leaf("team-red", &[], &[]);
        let no_san = der_of(&no_san);
        assert_eq!(identity_of(&no_san, TlsClientIdentity::SanUri), None);
        assert_eq!(identity_of(&no_san, TlsClientIdentity::SanDns), None);
        assert_eq!(
            identity_of(&no_san, TlsClientIdentity::Cn).as_deref(),
            Some("team-red")
        );

        let (long, _) = pki.leaf(
            "x",
            &[&format!("spiffe://{}", "a".repeat(MAX_TENANT_ID_LEN))],
            &[],
        );
        assert_eq!(
            identity_of(&der_of(&long), TlsClientIdentity::SanUri),
            None,
            "an over-long identity is no tenant key"
        );
        assert_eq!(
            identity_of(b"not a certificate", TlsClientIdentity::Cn),
            None
        );
    }

    /// A common name is read whatever ASN.1 string type its issuer used (a
    /// BMPString one is UTF-16, which the raw bytes would pass off as ASCII
    /// with interleaved NULs); a padded identity or one with control
    /// characters is none at all, since it is embedded verbatim in the
    /// tenant key, as a padded tenant id is no tenant key.
    #[test]
    fn a_common_name_is_decoded_from_its_string_type_and_bounded_like_a_tenant_id() {
        let pki = Pki::new();
        // OpenSSL takes the bytes of a typed entry verbatim: the UTF-16BE
        // spelling of the name, a NUL before each of its ASCII bytes.
        let utf16be: String = "team-red".chars().flat_map(|c| ['\0', c]).collect();
        let (bmp, _) = pki.leaf_named(&name_with_type(&utf16be, Asn1Type::BMPSTRING), &[], &[]);
        assert_eq!(
            identity_of(&der_of(&bmp), TlsClientIdentity::Cn).as_deref(),
            Some("team-red")
        );
        let (padded, _) = pki.leaf(" team-red ", &[], &[]);
        assert_eq!(
            identity_of(&der_of(&padded), TlsClientIdentity::Cn),
            None,
            "a padded value is no tenant key"
        );
        let (control, _) = pki.leaf("team\u{7}red", &[], &[]);
        assert_eq!(
            identity_of(&der_of(&control), TlsClientIdentity::Cn),
            None,
            "a control character is no part of a tenant key"
        );
        let (padded_uri, _) = pki.leaf(
            "team-red",
            &[" spiffe://cluster.local/ns/default/sa/x"],
            &[],
        );
        assert_eq!(
            identity_of(&der_of(&padded_uri), TlsClientIdentity::SanUri),
            None,
            "the bound applies to every field"
        );
    }

    /// A verifier needs usable material: a bundle without a CA, a key file
    /// without a key, and the off mode are refused by name.
    #[test]
    fn the_configuration_names_what_is_wrong_with_its_material() {
        let pki = Pki::new();
        let (cert, key) = pki.leaf("localhost", &[], &["localhost"]);

        assert!(
            rustls_config_with_client_auth(&cert, &key, &pki.ca_pem, TlsClientAuth::Required)
                .is_ok()
        );
        assert!(
            rustls_config_with_client_auth(&cert, &key, &pki.ca_pem, TlsClientAuth::Optional)
                .is_ok()
        );
        let err = rustls_config_with_client_auth(&cert, &key, b"not pem", TlsClientAuth::Required)
            .unwrap_err();
        assert!(err.contains("client CA bundle"), "{err}");
        let err =
            rustls_config_with_client_auth(&cert, b"not pem", &pki.ca_pem, TlsClientAuth::Required)
                .unwrap_err();
        assert!(err.contains("server key"), "{err}");
        let err = rustls_config_with_client_auth(&cert, &key, &pki.ca_pem, TlsClientAuth::Off)
            .unwrap_err();
        assert!(err.contains("off"), "{err}");
    }

    /// The service hands the connection's certificate to every request, and
    /// nothing to the requests of a connection without one.
    #[tokio::test]
    async fn the_service_attaches_the_certificate_to_each_request() {
        use tower::ServiceExt;

        let echo = tower::service_fn(|request: Request<Body>| async move {
            let body = match request.extensions().get::<ClientCertificate>() {
                Some(ClientCertificate::Identified(identity)) => identity.to_string(),
                Some(ClientCertificate::Unidentified) => "unidentified".to_string(),
                None => "none".to_string(),
            };
            Ok::<_, std::convert::Infallible>(Response::new(body))
        });
        let identified = IdentityService {
            inner: echo,
            certificate: Some(ClientCertificate::Identified(Arc::from(SPIFFE))),
        };
        for _ in 0..2 {
            let response = identified
                .clone()
                .oneshot(Request::new(Body::empty()))
                .await
                .unwrap();
            assert_eq!(response.into_body(), SPIFFE);
        }
        let bare = IdentityService {
            inner: echo,
            certificate: None,
        };
        let response = bare.oneshot(Request::new(Body::empty())).await.unwrap();
        assert_eq!(response.into_body(), "none");
    }

    /// One HTTPS GET of `/` over a fresh connection to `addr`: the status
    /// line and body, or the handshake's error.
    async fn https_get(
        addr: std::net::SocketAddr,
        roots: &[u8],
        identity: Option<(&[u8], &[u8])>,
    ) -> Result<String, String> {
        let mut store = RootCertStore::empty();
        store.add_parsable_certificates(
            rustls_pemfile::certs(&mut &*roots)
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
        );
        let builder = ClientConfig::builder().with_root_certificates(store);
        let config = match identity {
            Some((cert, key)) => builder
                .with_client_auth_cert(
                    rustls_pemfile::certs(&mut &*cert)
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap(),
                    rustls_pemfile::private_key(&mut &*key).unwrap().unwrap(),
                )
                .unwrap(),
            None => builder.with_no_client_auth(),
        };
        let connector = TlsConnector::from(Arc::new(config));
        let tcp = TcpStream::connect(addr).await.map_err(|e| e.to_string())?;
        let mut tls = connector
            .connect(ServerName::try_from("localhost").unwrap(), tcp)
            .await
            .map_err(|e| e.to_string())?;
        tls.write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .await
            .map_err(|e| e.to_string())?;
        let mut response = Vec::new();
        // The server closes after the response; a close_notify the client did
        // not read first is reported as an error by some stacks.
        let _ = tls.read_to_end(&mut response).await;
        String::from_utf8(response).map_err(|e| e.to_string())
    }

    /// A gateway-shaped TLS listener: the acceptor chain of the HTTPS
    /// listener over an echo route, answering the certificate's identity.
    #[expect(
        clippy::disallowed_methods,
        reason = "the server lives exactly as long as the test"
    )]
    fn serve(config: RustlsConfig, field: TlsClientIdentity) -> std::net::SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let app = Router::new().route(
            "/",
            get(|request: Request<Body>| async move {
                match request.extensions().get::<ClientCertificate>() {
                    Some(ClientCertificate::Identified(identity)) => identity.to_string(),
                    Some(ClientCertificate::Unidentified) => "unidentified".to_string(),
                    None => "none".to_string(),
                }
            }),
        );
        tokio::spawn(async move {
            axum_server::from_tcp_rustls(listener, config)
                .unwrap()
                .map(|acceptor| ClientIdentityAcceptor::new(acceptor, Some(field)))
                .serve(app.into_make_service())
                .await
                .unwrap();
        });
        addr
    }

    /// Over a real handshake: under `required` a client with a certificate
    /// from the CA is identified by it and a client without one does not get
    /// through; under `optional` the bare client is served without an
    /// identity; a certificate from another CA is refused.
    #[tokio::test]
    async fn the_handshake_verifies_and_the_connection_carries_the_identity() {
        crate::tls::install_crypto_provider();
        let pki = Pki::new();
        let (server_cert, server_key) = pki.leaf("localhost", &[], &["localhost"]);
        let (client_cert, client_key) = pki.leaf("team-red", &[SPIFFE], &["team-red.example"]);
        let other = Pki::new();
        let (foreign_cert, foreign_key) = other.leaf("intruder", &[SPIFFE], &[]);

        let required = serve(
            rustls_config_with_client_auth(
                &server_cert,
                &server_key,
                &pki.ca_pem,
                TlsClientAuth::Required,
            )
            .unwrap(),
            TlsClientIdentity::SanUri,
        );
        let optional = serve(
            rustls_config_with_client_auth(
                &server_cert,
                &server_key,
                &pki.ca_pem,
                TlsClientAuth::Optional,
            )
            .unwrap(),
            TlsClientIdentity::SanDns,
        );

        tokio::time::timeout(Duration::from_secs(30), async {
            let identified = https_get(required, &pki.ca_pem, Some((&client_cert, &client_key)))
                .await
                .unwrap();
            assert!(identified.starts_with("HTTP/1.1 200"), "{identified}");
            assert!(identified.ends_with(SPIFFE), "{identified}");

            let bare = https_get(required, &pki.ca_pem, None).await;
            assert!(
                !bare.as_deref().is_ok_and(|r| r.starts_with("HTTP/1.1")),
                "a client without a certificate is not served under required: {bare:?}"
            );

            let foreign =
                https_get(required, &pki.ca_pem, Some((&foreign_cert, &foreign_key))).await;
            assert!(
                !foreign.as_deref().is_ok_and(|r| r.starts_with("HTTP/1.1")),
                "a certificate from another CA is not served: {foreign:?}"
            );

            let by_dns = https_get(optional, &pki.ca_pem, Some((&client_cert, &client_key)))
                .await
                .unwrap();
            assert!(by_dns.ends_with("team-red.example"), "{by_dns}");

            let served_bare = https_get(optional, &pki.ca_pem, None).await.unwrap();
            assert!(served_bare.starts_with("HTTP/1.1 200"), "{served_bare}");
            assert!(served_bare.ends_with("none"), "{served_bare}");
        })
        .await
        .expect("the handshakes complete in time");
    }
}
