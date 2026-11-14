//! The media connector's HTTP client: the one construction through which the
//! router's own processing and the pipeline run worker-side fetch the media a
//! request names by URL.
//!
//! # Why idle connections are kept only briefly
//!
//! A media host closes an idle keep-alive connection on its own schedule,
//! often within seconds. A fetch that takes a pooled connection while the
//! host is closing it is written to a dead connection and fails before any
//! response byte. The fetch is not retried, so the request fails for a reason
//! it does not carry. The HTTP client's default keeps an idle connection for
//! 90 s, longer than many hosts do; `POOL_IDLE_TIMEOUT` keeps it for 3 s.

use std::time::Duration;

use anyhow::Result;
use llm_multimodal::{MediaConnector, MediaConnectorConfig};

use crate::worker::http_client::build_with;

/// The client's bound on one request, from connecting to the last byte of the
/// body. A fetch with a budget of its own (`fetch_timeout`,
/// `video_fetch_timeout`) runs under that budget instead.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// How long the media client keeps an idle connection for reuse.
///
/// The worker clients' default (`upstream_pool_idle_timeout_secs`), for the
/// race that setting documents: the bound must run out before the server's
/// keep-alive does, or a checkout meets the server's close. It is a constant
/// and not that setting, which an operator fits to the engines they run: the
/// media hosts are whichever ones requests name, and no setting can know
/// their keep-alive. A host that closes sooner than this can still close a
/// connection under a fetch. The bound narrows that window from the client's
/// default of 90 s; it does not remove it.
const POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(3);

/// The media connector over `config`, fetching through an HTTP client that
/// bounds a request by `REQUEST_TIMEOUT` and an idle connection by
/// `POOL_IDLE_TIMEOUT`. A host without a native CA root store is tolerated as
/// [`build_with`] describes.
pub(super) fn media_connector(config: &MediaConnectorConfig) -> Result<MediaConnector> {
    media_connector_with_pool_idle_timeout(config, POOL_IDLE_TIMEOUT)
}

/// [`media_connector`] with the idle bound given: the tests' way to a bound
/// short enough to wait out.
fn media_connector_with_pool_idle_timeout(
    config: &MediaConnectorConfig,
    pool_idle_timeout: Duration,
) -> Result<MediaConnector> {
    build_with(
        || {
            crate::server::http_client_builder()
                .timeout(REQUEST_TIMEOUT)
                .pool_idle_timeout(pool_idle_timeout)
        },
        false,
        "media HTTP client",
        |client| MediaConnector::new(client, config.clone()),
    )
    .map_err(anyhow::Error::msg)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use llm_multimodal::{ImageFetchConfig, MediaSource};
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::{TcpListener, TcpStream},
        sync::watch,
    };

    use super::*;

    /// The longest a test waits for anything: ample for a loaded host, and a
    /// wait that outlasts it fails the test instead of hanging it.
    const GUARD: Duration = Duration::from_secs(10);

    /// A loopback media host that answers every `GET` with the same PNG and
    /// keeps each connection open until the client closes it.
    struct Origin {
        url: String,
        /// Connections accepted so far.
        accepted: watch::Receiver<usize>,
        /// Connections the client has closed so far.
        closed: watch::Receiver<usize>,
    }

    impl Origin {
        async fn start() -> Self {
            let mut png = std::io::Cursor::new(Vec::new());
            image::DynamicImage::ImageRgb8(image::RgbImage::new(8, 8))
                .write_to(&mut png, image::ImageFormat::Png)
                .unwrap();
            let png: Arc<[u8]> = png.into_inner().into();

            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}/image.png", listener.local_addr().unwrap());
            let (accepted_tx, accepted) = watch::channel(0);
            let (closed_tx, closed) = watch::channel(0);
            #[expect(
                clippy::disallowed_methods,
                reason = "the test's media host lives as long as the test's runtime, which ends it"
            )]
            tokio::spawn(async move {
                while let Ok((socket, _)) = listener.accept().await {
                    accepted_tx.send_modify(|accepted| *accepted += 1);
                    let (png, closed_tx) = (Arc::clone(&png), closed_tx.clone());
                    #[expect(
                        clippy::disallowed_methods,
                        reason = "one task per connection, ended with the test's runtime"
                    )]
                    tokio::spawn(async move {
                        answer_until_closed(socket, &png).await;
                        closed_tx.send_modify(|closed| *closed += 1);
                    });
                }
            });
            Self {
                url,
                accepted,
                closed,
            }
        }

        fn accepted(&self) -> usize {
            *self.accepted.borrow()
        }

        fn closed(&self) -> usize {
            *self.closed.borrow()
        }
    }

    /// Answers every `GET` on `socket` with `body` and returns when the
    /// client closes the connection. A `GET` carries no body, so the blank
    /// line ends each request.
    async fn answer_until_closed(mut socket: TcpStream, body: &[u8]) {
        let head = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: image/png\r\ncontent-length: {}\r\n\r\n",
            body.len()
        );
        let mut request = Vec::new();
        let mut chunk = [0u8; 1024];
        loop {
            match socket.read(&mut chunk).await {
                Ok(0) | Err(_) => return,
                Ok(read) => request.extend_from_slice(&chunk[..read]),
            }
            if request.ends_with(b"\r\n\r\n") {
                request.clear();
                if socket.write_all(head.as_bytes()).await.is_err()
                    || socket.write_all(body).await.is_err()
                {
                    return;
                }
            }
        }
    }

    /// The gateway's media connector, let through to the loopback host, with
    /// an idle connection kept for `pool_idle_timeout`.
    fn connector(pool_idle_timeout: Duration) -> MediaConnector {
        let config = MediaConnectorConfig {
            allow_private_hosts: true,
            ..MediaConnectorConfig::default()
        };
        media_connector_with_pool_idle_timeout(&config, pool_idle_timeout).unwrap()
    }

    async fn fetch(connector: &MediaConnector, origin: &Origin) {
        let source = MediaSource::Url(origin.url.clone());
        tokio::time::timeout(
            GUARD,
            connector.fetch_image(source, ImageFetchConfig::default()),
        )
        .await
        .expect("the loopback host answers within the guard")
        .expect("the image is fetched");
    }

    /// A connection the client gave up by itself is one the host cannot close
    /// under a fetch: past the bound the client closes its idle connection,
    /// and the next fetch connects anew.
    #[tokio::test]
    async fn a_fetch_after_the_idle_bound_arrives_on_a_new_connection() {
        let mut origin = Origin::start().await;
        let connector = connector(Duration::from_millis(100));

        fetch(&connector, &origin).await;
        assert_eq!(origin.accepted(), 1, "the first fetch connects once");

        // The host sees the client's close as an event, so the test waits
        // for that and never sleeps the bound out.
        tokio::time::timeout(GUARD, origin.closed.wait_for(|&closed| closed == 1))
            .await
            .expect("the client closes its idle connection once the bound has passed")
            .unwrap();

        fetch(&connector, &origin).await;
        assert_eq!(
            origin.accepted(),
            2,
            "the fetch after the bound arrives on a connection of its own"
        );
    }

    /// The bound ends the reuse of a stale connection, not reuse: fetches
    /// that follow each other inside it share one connection.
    #[tokio::test]
    async fn fetches_inside_the_idle_bound_share_one_connection() {
        let origin = Origin::start().await;
        // Longer than any run of this test, so the second fetch is inside the
        // bound however slow the host.
        let connector = connector(Duration::from_secs(600));

        fetch(&connector, &origin).await;
        fetch(&connector, &origin).await;
        assert_eq!(
            origin.accepted(),
            1,
            "the second fetch reuses the first one's connection"
        );
        assert_eq!(origin.closed(), 0, "the connection is still pooled");
    }
}
