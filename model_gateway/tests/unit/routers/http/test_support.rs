//! Stub upstreams the HTTP routers' tests share.

use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Whether `bytes` hold a whole HTTP/1 request: the head and as many
/// body bytes as its `Content-Length` declares.
pub(crate) fn request_complete(bytes: &[u8]) -> bool {
    let Some(head_end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") else {
        return false;
    };
    let head = String::from_utf8_lossy(&bytes[..head_end]);
    let length = head
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.trim()
                .eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())?
        })
        .unwrap_or(0);
    bytes.len() >= head_end + 4 + length
}

/// A worker that answers one request with a sized SSE head (the
/// `Content-Length` of the whole `body`) and closes the connection after
/// the first `sent` bytes of it: an upstream that knew its length and
/// went away mid-body.
#[expect(
    clippy::disallowed_methods,
    reason = "test stub server lives for the duration of the test process"
)]
pub(crate) async fn spawn_sized_sse_worker_cut_after(body: &'static str, sent: usize) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let Ok((mut sock, _)) = listener.accept().await else {
            return;
        };
        // The whole request first, so the close is a clean end of the
        // answer and never a reset of unread request bytes.
        let mut request = Vec::new();
        let mut buf = [0u8; 4096];
        while !request_complete(&request) {
            match sock.read(&mut buf).await {
                Ok(n) if n > 0 => request.extend_from_slice(&buf[..n]),
                _ => return,
            }
        }
        let head = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\r\n",
            body.len()
        );
        let _ = sock.write_all(head.as_bytes()).await;
        let _ = sock.write_all(&body.as_bytes()[..sent]).await;
        let _ = sock.flush().await;
        // The socket drops here, short of the declared length.
    });
    format!("http://{addr}")
}
