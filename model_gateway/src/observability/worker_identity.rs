//! Public worker identities for metrics and circuit-breaker logs.

use std::sync::{Arc, OnceLock};

use rand::RngExt;

use crate::worker::endpoint::Endpoint;

/// Preserve public endpoint spellings; hide sensitive ones behind a process-local identity.
///
/// Worker endpoint grammar defines a final numeric `@rank` suffix, including
/// `grpc://name@1`. Parse that suffix before checking the remaining authority
/// for userinfo. Never normalize the returned clean label: existing dashboards
/// depend on the original spelling. The keyed digest keeps different private
/// endpoints distinct without allowing offline guesses of their credentials.
/// No raw URL is retained globally; only the random process key is retained.
pub(crate) fn public_worker_label(url: &str) -> Arc<str> {
    let public = !url.contains(['?', '#'])
        && Endpoint::parse_with_rank(url).is_ok_and(|(endpoint, _)| {
            if endpoint.is_ipc() {
                return true;
            }
            let rendered = endpoint.render();
            let authority = rendered
                .split_once("://")
                .map_or(rendered.as_str(), |(_, rest)| rest);
            !authority
                .split('/')
                .next()
                .unwrap_or_default()
                .contains('@')
        });
    if public {
        return Arc::from(url);
    }

    static KEY: OnceLock<[u8; 32]> = OnceLock::new();
    let key = KEY.get_or_init(|| rand::rng().random());
    Arc::from(format!(
        "worker:{}",
        blake3::keyed_hash(key, url.as_bytes()).to_hex()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_worker_endpoints_keep_exact_labels() {
        for url in [
            "http://localhost:8080/",
            "grpc://cluster.local@1",
            "http://localhost:8080@2",
            "ipc:///tmp/worker.sock@3",
            "http://[::1]:8080",
            "http://localhost:8080/path@text",
            "ipc:///tmp/user@worker.sock",
        ] {
            assert_eq!(&*public_worker_label(url), url);
        }
    }

    #[test]
    fn sensitive_worker_endpoints_are_opaque_stable_and_distinct() {
        let urls = [
            "http://alice:password@localhost:8080",
            "http://alice:other-password@localhost:8080",
            "http://alice@localhost:8080@2",
            "http://localhost:8080?token=secret",
            "http://localhost:8080#secret",
            "http://@",
            "http://[invalid]:8080",
            "http://alice%40example:password%21@localhost:8080",
            "http://alice:password@localhost:8080@2",
        ];
        let labels: Vec<_> = urls.iter().map(|url| public_worker_label(url)).collect();
        for (url, label) in urls.iter().zip(&labels) {
            assert!(
                label.starts_with("worker:"),
                "sensitive endpoint must be opaque"
            );
            assert!(!label.contains("alice"));
            assert!(!label.contains("password"));
            assert!(!label.contains("secret"));
            assert_eq!(label, &public_worker_label(url));
        }
        let unique: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(unique.len(), urls.len());
    }
}
