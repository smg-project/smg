//! Common test utilities for the multimodal integration tests.
#![allow(dead_code, clippy::allow_attributes)] // each test binary uses a subset of these helpers

use std::{io::Read, net::TcpStream};

use rustls::crypto::ring;

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
