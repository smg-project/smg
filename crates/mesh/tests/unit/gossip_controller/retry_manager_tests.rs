use super::*;

#[test]
fn first_retry_uses_initial_delay() {
    let mut mgr = RetryManager::default();
    mgr.record_attempt();
    // ExponentialBackoff::default() = (1s, 60s, 2.0)
    assert_eq!(mgr.next_delay(), Duration::from_secs(1));
}

#[test]
fn subsequent_retries_double_until_capped() {
    let mut mgr = RetryManager::default();
    mgr.record_attempt();
    assert_eq!(mgr.next_delay(), Duration::from_secs(1));
    mgr.record_attempt();
    assert_eq!(mgr.next_delay(), Duration::from_secs(2));
    mgr.record_attempt();
    assert_eq!(mgr.next_delay(), Duration::from_secs(4));
    // Cap is 60s with the default config: 1 * 2^6 = 64 -> clamped.
    for _ in 0..10 {
        mgr.record_attempt();
    }
    assert_eq!(mgr.next_delay(), Duration::from_secs(60));
}

#[test]
fn reset_returns_to_first_retry_state() {
    let mut mgr = RetryManager::default();
    for _ in 0..5 {
        mgr.record_attempt();
    }
    assert_ne!(mgr.next_delay(), Duration::from_secs(1));
    mgr.reset();
    assert!(mgr.should_retry(), "post-reset should always allow retry");
    mgr.record_attempt();
    assert_eq!(mgr.next_delay(), Duration::from_secs(1));
}

#[test]
fn should_retry_before_any_attempt() {
    let mgr = RetryManager::default();
    assert!(mgr.should_retry());
}
