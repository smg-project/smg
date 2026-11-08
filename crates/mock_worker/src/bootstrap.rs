//! The bootstrap port a prefill or encode worker binds, on either wire: where
//! a partner's engine would rendezvous for the handoff, which the mock never
//! performs (the decode side has the prompt). Each connection is accepted and
//! closed, which is all the gateway's probe of the port asks, until the
//! worker stops.

use std::time::Duration;

use tokio::{net::TcpListener, sync::watch};

/// Bind `port` on `host` for the worker that advertises it, before the worker
/// serves, as an engine's bootstrap server is up when its health answers.
/// `None` for a worker that binds no port; an error names a port that could
/// not be bound, which stops the worker as its own port would.
pub(crate) async fn bind(host: &str, port: Option<u16>) -> Result<Option<TcpListener>, String> {
    let Some(port) = port else {
        return Ok(None);
    };
    TcpListener::bind((host, port))
        .await
        .map(Some)
        .map_err(|e| format!("bootstrap port bind {host}:{port} failed: {e}"))
}

/// Accept and close every connection to the bootstrap port until the worker
/// stops.
pub(crate) async fn accept_and_close(listener: TcpListener, mut stopped: watch::Receiver<bool>) {
    loop {
        tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok((connection, _)) => drop(connection),
                Err(e) => {
                    tracing::warn!(
                        "bootstrap port {:?}: accept failed: {e}",
                        listener.local_addr().ok()
                    );
                    // A listener out of descriptors would fail every accept
                    // at once; a pause keeps the loop from spinning on it.
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            },
            // The wait's guard stays inside its own future: a select arm
            // that yielded it would hold it across the loop's awaits.
            () = async {
                let _ = stopped.wait_for(|stop| *stop).await;
            } => break,
        }
    }
}
