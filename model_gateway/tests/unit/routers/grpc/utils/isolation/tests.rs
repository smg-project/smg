use super::*;

#[test]
fn a_slice_panics_quoted_bytes_are_elided() {
    assert_eq!(
        sanitized("byte index 7 is out of bounds of `secret text`"),
        "byte index 7 is out of bounds of `...`"
    );
    assert_eq!(
        sanitized("begin <= end (4 <= 2) when slicing `abc`[...]"),
        "begin <= end (4 <= 2) when slicing `...`[...]"
    );
    assert_eq!(
        sanitized("byte index 1 is not a char boundary; it is inside 'é' (bytes 0..2) of `é`"),
        "byte index 1 is not a char boundary; it is inside 'é' (bytes 0..2) of `...`"
    );
    // An opening backtick that never closes cuts the rest.
    assert_eq!(sanitized("the output was `abc"), "the output was `...`");
    // A message with no quoted span is itself.
    assert_eq!(
        sanitized("index out of bounds: the len is 3 but the index is 5"),
        "index out of bounds: the len is 3 but the index is 5"
    );
}

#[test]
fn a_long_message_is_cut_on_a_character_boundary() {
    let long = "é".repeat(200);
    let shown = sanitized(&long);
    assert!(shown.ends_with("..."), "{shown}");
    assert!(shown.len() <= MESSAGE_SHOWN + 3, "{}", shown.len());
    assert!(shown.trim_end_matches("...").chars().all(|c| c == 'é'));
}

#[test]
fn a_panics_payload_of_either_text_kind_is_its_message() {
    let caught: Result<(), _> = isolated_sync(|| std::panic::panic_any("a static message"));
    assert_eq!(caught.expect_err("the panic").message(), "a static message");
    let caught: Result<(), _> = isolated_sync(|| std::panic::panic_any(format!("built {}", 1)));
    assert_eq!(caught.expect_err("the panic").message(), "built 1");
    let caught: Result<(), _> = isolated_sync(|| std::panic::panic_any(7_u32));
    assert_eq!(caught.expect_err("the panic").message(), "unknown panic");
    // The test trap's own panic quotes the marker the way a slice panic
    // quotes its bytes, so the line elides it like any quoted span.
    let caught: Result<(), _> = isolated_sync(|| spring_test_trap("MARK"));
    let sprung = caught.expect_err("the trap");
    assert_eq!(
        sprung.message(),
        "the test trap sprang: the response carried `...`"
    );
}

/// The guard's future is the reading's future plus a few words, never a
/// second copy of it: the streaming paths spawn the guarded reading as
/// a task, and the task's allocation is the future's size.
#[test]
fn the_guard_holds_the_readings_future_once() {
    use std::mem::size_of_val;
    async fn reading(state: [u8; 4096]) -> usize {
        tokio::task::yield_now().await;
        state.len()
    }
    let bare = size_of_val(&reading([0; 4096]));
    let guarded = size_of_val(&isolated(reading([0; 4096])));
    assert!(
        guarded <= bare + 64,
        "the guard's future is {guarded} bytes around a reading of {bare}"
    );
}

#[tokio::test]
async fn a_future_that_panics_is_the_requests_failure_and_one_that_does_not_is_its_output() {
    let fine = isolated(async { 41 + 1 }).await;
    assert_eq!(fine.expect("the output"), 42);
    let caught = isolated(async {
        tokio::task::yield_now().await;
        let bytes = "model output";
        // A slice panic of the kind a reader has: whatever the standard
        // library's message quotes of the string, none of it is kept.
        let _ = &bytes[..usize::MAX];
    })
    .await;
    let panicked = caught.expect_err("the panic").read_by(Some("qwen3"));
    assert!(
        panicked.message().contains("out of bounds"),
        "{}",
        panicked.message()
    );
    assert!(
        !panicked.message().contains("model output"),
        "{}",
        panicked.message()
    );
    // The client's text names the failure alone; the message is the log
    // line's.
    assert_eq!(
        panicked.to_string(),
        "the gateway's reader of the response panicked"
    );
}
