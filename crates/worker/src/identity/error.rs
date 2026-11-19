//! The worker errors: what a worker operation reports when it fails, the
//! worker's address redacted where the message names it (its credentials
//! masked, a query or fragment dropped, an unparsable one replaced by a
//! marker; a plain address passes through).

use super::endpoint::Endpoint;

/// Worker-related errors. An arm that names a worker writes its address
/// redacted ([`Endpoint::redacted_url`]: the credentials masked, a query
/// or fragment dropped, a plain address byte for byte, an unparsable one
/// replaced by a marker): the message follows a redacted field into log
/// lines, step failures and response bodies, and must not repeat the
/// address raw. The `url` field keeps the registered address for the code
/// that matches on it.
#[derive(Debug, thiserror::Error)]
pub enum WorkerError {
    #[error("Health check failed for worker {}: {reason}", shown(.url))]
    HealthCheckFailed { url: String, reason: String },

    #[error("Worker not found: {}", shown(.url))]
    WorkerNotFound { url: String },

    #[error("Invalid worker configuration: {message}")]
    InvalidConfiguration { message: String },

    #[error("Network error for worker {}: {error}", shown(.url))]
    NetworkError { url: String, error: String },

    #[error("Connection failed for worker {}: {reason}", shown(.url))]
    ConnectionFailed { url: String, reason: String },

    #[error("{operation} failed for worker {}: {reason}", shown(.url))]
    OperationFailed {
        url: String,
        operation: String,
        reason: String,
    },
}

/// The address as an error message writes it: masked when it carries
/// userinfo, cut at a query or fragment, a marker when the parser refuses
/// it; an error without one (a network error whose request had no URL)
/// writes nothing where the address goes.
fn shown(url: &str) -> String {
    if url.is_empty() {
        String::new()
    } else {
        Endpoint::redacted_url(url)
    }
}

/// Result type for worker operations
pub type WorkerResult<T> = Result<T, WorkerError>;

/// A client error as a worker error. The client's text would repeat the
/// request URL as the client holds it, so the URL moves to `url` (kept
/// raw; the message writes it redacted) and the text goes without it.
impl From<reqwest::Error> for WorkerError {
    fn from(err: reqwest::Error) -> Self {
        let url = err.url().map(|u| u.to_string()).unwrap_or_default();
        WorkerError::NetworkError {
            url,
            error: err.without_url().to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        error::Error,
        net::{Ipv4Addr, SocketAddr},
    };

    use tokio::net::TcpSocket;

    use super::*;
    use crate::connection::http_client::LazyHttpClient;

    #[tokio::test]
    async fn a_client_error_converts_without_the_clients_url() {
        // Bound and never listening: the host's own stack refuses every
        // connection, and the port stays taken while the socket lives.
        let refused = TcpSocket::new_v4().expect("a TCP socket");
        refused
            .bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .expect("a loopback port of the system's choice");
        let addr = refused.local_addr().expect("the bound address");
        let url = format!("http://{addr}/health?token=s3cret");
        let error: WorkerError = LazyHttpClient::deferred()
            .client()
            .get(&url)
            .send()
            .await
            .expect_err("nothing listens at the address")
            .into();
        let text = error.to_string();
        assert!(!text.contains("s3cret"), "{text}");
        assert_eq!(text.matches(&addr.to_string()).count(), 1, "{text}");
    }

    #[test]
    fn test_health_check_failed_display() {
        let error = WorkerError::HealthCheckFailed {
            url: "http://worker1:8080".to_string(),
            reason: "Connection refused".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Health check failed for worker http://worker1:8080: Connection refused"
        );
    }

    #[test]
    fn test_worker_not_found_display() {
        let error = WorkerError::WorkerNotFound {
            url: "http://worker2:8080".to_string(),
        };
        assert_eq!(error.to_string(), "Worker not found: http://worker2:8080");
    }

    #[test]
    fn test_invalid_configuration_display() {
        let error = WorkerError::InvalidConfiguration {
            message: "Missing port number".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Invalid worker configuration: Missing port number"
        );
    }

    #[test]
    fn test_network_error_display() {
        let error = WorkerError::NetworkError {
            url: "http://worker3:8080".to_string(),
            error: "Timeout after 30s".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Network error for worker http://worker3:8080: Timeout after 30s"
        );
    }

    #[test]
    fn test_worker_error_implements_std_error() {
        let error = WorkerError::WorkerNotFound {
            url: "http://test".to_string(),
        };
        let _: &dyn Error = &error;
        assert!(error.source().is_none());
    }

    #[test]
    fn test_error_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<WorkerError>();
    }

    #[test]
    fn test_worker_result_type_alias() {
        let result: WorkerResult<i32> = Ok(42);
        assert!(matches!(result, Ok(42)));

        let error = WorkerError::WorkerNotFound {
            url: "test".to_string(),
        };
        let result: WorkerResult<i32> = Err(error);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_url_handling() {
        let error1 = WorkerError::HealthCheckFailed {
            url: String::new(),
            reason: "No connection".to_string(),
        };
        assert_eq!(
            error1.to_string(),
            "Health check failed for worker : No connection"
        );

        let error2 = WorkerError::NetworkError {
            url: String::new(),
            error: "DNS failure".to_string(),
        };
        assert_eq!(error2.to_string(), "Network error for worker : DNS failure");

        let error3 = WorkerError::WorkerNotFound { url: String::new() };
        assert_eq!(error3.to_string(), "Worker not found: ");
    }

    #[test]
    fn test_special_characters_in_messages() {
        let error = WorkerError::InvalidConfiguration {
            message: "Invalid JSON: {\"error\": \"test\"}".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Invalid worker configuration: Invalid JSON: {\"error\": \"test\"}"
        );

        let error2 = WorkerError::HealthCheckFailed {
            url: "http://测试:8080".to_string(),
            reason: "连接被拒绝".to_string(),
        };
        assert_eq!(
            error2.to_string(),
            "Health check failed for worker http://测试:8080: 连接被拒绝"
        );
    }

    #[test]
    fn test_very_long_error_messages() {
        let long_message = "A".repeat(10000);
        let error = WorkerError::InvalidConfiguration {
            message: long_message.clone(),
        };
        let display = error.to_string();
        assert!(display.contains(&long_message));
        assert_eq!(
            display.len(),
            "Invalid worker configuration: ".len() + long_message.len()
        );
    }

    #[test]
    fn test_reqwest_error_conversion() {
        let network_error = WorkerError::NetworkError {
            url: "http://example.com".to_string(),
            error: "connection timeout".to_string(),
        };

        match network_error {
            WorkerError::NetworkError { url, error } => {
                assert_eq!(url, "http://example.com");
                assert_eq!(error, "connection timeout");
            }
            _ => panic!("Expected NetworkError variant"),
        }
    }

    #[test]
    fn a_credentialed_address_is_masked_in_every_arm_that_names_one() {
        let url = "http://user:secret@worker:3000".to_string();
        let errors = [
            WorkerError::HealthCheckFailed {
                url: url.clone(),
                reason: "health probe returned non-success".to_string(),
            },
            WorkerError::WorkerNotFound { url: url.clone() },
            WorkerError::NetworkError {
                url: url.clone(),
                error: "timeout".to_string(),
            },
            WorkerError::ConnectionFailed {
                url: url.clone(),
                reason: "refused".to_string(),
            },
            WorkerError::OperationFailed {
                url: url.clone(),
                operation: "Flush cache".to_string(),
                reason: "500".to_string(),
            },
        ];
        for error in &errors {
            let line = error.to_string();
            assert!(
                !line.contains("secret") && line.contains("http://***@worker:3000"),
                "{line}"
            );
        }
        // The field keeps the registered address for the code that matches on it.
        assert!(
            matches!(&errors[1], WorkerError::WorkerNotFound { url } if url.contains("secret"))
        );
        // A query may carry a token: cut, as a log line cuts it.
        let error = WorkerError::ConnectionFailed {
            url: "http://worker:3000/v1?token=secret".to_string(),
            reason: "refused".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Connection failed for worker http://worker:3000/v1: refused"
        );
    }

    #[test]
    fn test_error_equality() {
        let error1 = WorkerError::WorkerNotFound {
            url: "http://test".to_string(),
        };
        let error2 = WorkerError::WorkerNotFound {
            url: "http://test".to_string(),
        };
        assert_eq!(error1.to_string(), error2.to_string());
    }
}
