use super::*;

#[test]
fn test_backoff_calculation() {
    let manager = ReconnectionManager::new();
    assert_eq!(manager.calculate_backoff(1), Duration::from_millis(500));
    assert_eq!(manager.calculate_backoff(2), Duration::from_millis(1000));
    assert_eq!(manager.calculate_backoff(5), Duration::from_millis(8000));
}
