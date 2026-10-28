//! Common test utilities for the multimodal integration tests.
#![allow(dead_code, clippy::allow_attributes)] // each test binary uses a subset of these helpers

use std::{
    io::Read,
    net::{Ipv4Addr, SocketAddr, TcpStream},
};

use rustls::crypto::ring;
use tokio::net::TcpSocket;

/// The tests' HTTP client for the media connectors. The library takes its
/// client from the caller; the tests build it here, with the process-level
/// TLS crypto provider installed first (reqwest's `rustls-no-provider`
/// client panics at build time without one; a provider installed earlier in
/// the process stays).
#[expect(
    clippy::disallowed_methods,
    reason = "a test binary's client factory; the process TLS crypto provider is installed first"
)]
pub fn http_client() -> reqwest::Client {
    let _ = ring::default_provider().install_default();
    reqwest::Client::new()
}

/// A builder for a client with settings of its own, the provider installed
/// the same way.
#[expect(
    clippy::disallowed_methods,
    reason = "a test binary's client factory; the process TLS crypto provider is installed first"
)]
pub fn http_client_builder() -> reqwest::ClientBuilder {
    let _ = ring::default_provider().install_default();
    reqwest::Client::builder()
}

/// Reads a request's head off `stream`, up to and including the blank line,
/// as the tests' one-shot servers do before they answer.
pub fn read_head(stream: &mut TcpStream) {
    let mut head = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        let read = stream.read(&mut buf).unwrap_or(0);
        if read == 0 {
            break;
        }
        head.extend_from_slice(&buf[..read]);
        if head.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
    }
}

/// A loopback address nothing answers at, refused for as long as this value
/// lives (the shape of the gRPC client's `channel::testing::RefusedEndpoint`).
///
/// The socket behind it is bound and never listening: the host's own stack
/// refuses every connection at once, the way an unreachable peer's port
/// does, and the port stays taken, so no other test's `bind("127.0.0.1:0")`
/// can draw it in the meantime. A listener bound and dropped for the same
/// purpose gives its port back, and a neighbouring test's stub then answers
/// in the unreachable peer's place.
///
/// Keep the value in a named binding for the whole case: a `_` pattern or a
/// temporary drops it, and the port is free again.
#[must_use = "the address is refused only while this value lives"]
pub struct RefusedAddress {
    _reservation: TcpSocket,
    addr: SocketAddr,
}

impl RefusedAddress {
    /// Reserve a loopback port of the system's choice, and refuse it.
    #[allow(clippy::expect_used)] // a fixture: a loopback socket the system cannot give is a test failure
    pub fn reserve() -> Self {
        let socket = TcpSocket::new_v4().expect("a TCP socket");
        socket
            .bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .expect("a loopback port of the system's choice");
        let addr = socket.local_addr().expect("the bound address");
        Self {
            _reservation: socket,
            addr,
        }
    }

    /// The address, `127.0.0.1:<port>`.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// The port alone, for a URL built around it.
    pub fn port(&self) -> u16 {
        self.addr.port()
    }
}
