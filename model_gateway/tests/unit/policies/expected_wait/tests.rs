use super::ExpectedWait;

#[test]
fn combines_queued_dispatched_and_kv_wait() {
    let wait = ExpectedWait::new(800.0, 500.0, 0.5, 0.2);

    assert!((wait.seconds(200) - 2.2).abs() < f64::EPSILON);
}

#[test]
fn clamps_kv_usage_to_formula_bounds() {
    let below_zero = ExpectedWait::new(0.0, 100.0, -1.0, 0.2).seconds(0);
    let at_upper_bound = ExpectedWait::new(0.0, 100.0, 0.999, 0.2).seconds(0);
    let above_one = ExpectedWait::new(0.0, 100.0, 2.0, 0.2).seconds(0);

    assert_eq!(below_zero, 0.0);
    assert!(at_upper_bound.is_finite());
    assert_eq!(above_one, at_upper_bound);
}
