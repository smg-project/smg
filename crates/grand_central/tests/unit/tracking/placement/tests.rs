use super::Placement;

/// A kept placement travels with its request: to the task that streams
/// the response, and behind a pointer several threads share.
#[test]
fn a_kept_placement_crosses_threads_and_borrows_nothing() {
    fn travels<T: Send + Sync + 'static>() {}
    travels::<Placement<'static>>();
}
