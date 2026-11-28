use super::*;

#[test]
fn outside_a_request_s_scope_there_is_no_stamp() {
    assert!(RequestStarted::current().is_none());
}

#[tokio::test]
async fn inside_the_scope_the_stamp_is_the_instant_it_was_taken_at() {
    let at = Instant::now() - Duration::from_secs(3);
    let read = RequestStarted::at(at)
        .scope(async { RequestStarted::current() })
        .await;
    assert_eq!(read, Some(at));
    assert!(
        RequestStarted::current().is_none(),
        "the scope ends with the request's future"
    );
}

#[test]
fn inside_the_sync_scope_the_stamp_is_read_from_a_call() {
    let at = Instant::now() - Duration::from_secs(2);
    let read = RequestStarted::at(at).sync_scope(RequestStarted::current);
    assert_eq!(read, Some(at));
    assert!(
        RequestStarted::current().is_none(),
        "the scope ends with the call"
    );
}

#[tokio::test(start_paused = true)]
async fn the_stamp_is_on_the_runtime_s_clock() {
    let before = RequestStarted::now();
    tokio::time::advance(Duration::from_secs(5)).await;
    let after = RequestStarted::now();
    assert_eq!(
        after.instant().duration_since(before.instant()),
        Duration::from_secs(5)
    );
}

/// A wait counted from the entry, as the dispatch wait is.
const WAIT: Duration = Duration::from_secs(9);

#[test]
fn outside_a_scope_an_upload_wait_is_written_nowhere() {
    RequestStarted::upload_waited(Duration::from_secs(1));
    assert!(RequestStarted::carried().is_none());
}

#[tokio::test]
async fn an_upload_wait_is_taken_off_a_wait_counted_from_the_entry() {
    let entered = RequestStarted::now();
    let (before, first, second) = entered
        .scope(async {
            let before = RequestStarted::carried();
            RequestStarted::upload_waited(Duration::from_secs(3));
            let first = RequestStarted::carried();
            RequestStarted::upload_waited(Duration::from_secs(4));
            (before, first, RequestStarted::carried())
        })
        .await;
    let before = before.expect("the stamp inside the scope");
    assert_eq!(
        before.less_upload_wait(WAIT),
        WAIT,
        "a body that never waited takes nothing off"
    );
    let first = first.expect("the stamp inside the scope");
    assert_eq!(first.instant(), entered.instant(), "the entry stamp stays");
    assert_eq!(first.less_upload_wait(WAIT), Duration::from_secs(6));
    let second = second.expect("the stamp inside the scope");
    assert_eq!(
        second.less_upload_wait(WAIT),
        Duration::from_secs(2),
        "a second read's wait adds to the first"
    );
    assert_eq!(
        second.less_upload_wait(Duration::from_secs(5)),
        Duration::ZERO,
        "a wait shorter than the upload wait is zero"
    );
}

#[tokio::test]
async fn a_task_of_its_own_carries_the_upload_wait_and_a_new_stamp_holds_none() {
    let entered = RequestStarted::now();
    let carried = entered
        .scope(async {
            RequestStarted::upload_waited(Duration::from_secs(2));
            RequestStarted::carried()
        })
        .await
        .expect("the stamp inside the scope");
    // A task of its own, run under the carried stamp, counts as the
    // request's future does.
    let again = carried
        .scope(async { RequestStarted::carried() })
        .await
        .expect("the carried stamp");
    assert_eq!(again.instant(), entered.instant());
    assert_eq!(again.less_upload_wait(WAIT), Duration::from_secs(7));
    // A stamp taken anew (a later turn's) holds no upload wait.
    assert_eq!(RequestStarted::now().less_upload_wait(WAIT), WAIT);
}
