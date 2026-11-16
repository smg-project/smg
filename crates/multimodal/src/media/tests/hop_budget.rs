use super::super::*;

/// The admission of a hop is held to the fetch budget: an answer in time
/// is the answer, one that stalls past the budget's end is the budget's
/// end, and without a budget the answer is waited for.
#[tokio::test]
async fn an_admission_is_held_to_the_fetch_budget() {
    assert_eq!(within_budget(None, async { 7 }).await, Some(7));
    assert_eq!(
        within_budget(Some(Duration::from_secs(5)), async { 7 }).await,
        Some(7)
    );
    assert_eq!(
        within_budget(
            Some(Duration::from_millis(10)),
            std::future::pending::<i32>()
        )
        .await,
        None
    );
}
