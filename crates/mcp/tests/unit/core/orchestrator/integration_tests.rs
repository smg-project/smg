use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;

#[tokio::test]
async fn test_reconnection_logic_flow() {
    let _orchestrator = McpOrchestrator::new_test();
    let name = "test-server";

    // simulate a server in the registry
    let manager = ReconnectionManager {
        base_delay: Duration::from_millis(1),
        max_retries: 3,
        ..Default::default()
    };

    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_clone = Arc::clone(&attempts);

    let result = manager
        .reconnect(name, || {
            let count = attempts_clone.fetch_add(1, Ordering::SeqCst);
            async move {
                if count < 1 {
                    Err(McpError::Transport("Handshake failed".to_string()))
                } else {
                    Ok(())
                }
            }
        })
        .await;

    assert!(result.is_ok());
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
}
