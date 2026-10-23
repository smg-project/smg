//! A request's reading of its response, isolated from the task it runs in.
//!
//! The parsers that read a model's output (Symphony's reader, the tool and
//! reasoning parsers, the folds that turn their events into a message or a
//! frame) run over bytes the gateway does not control. A panic in one of
//! them used to be the task's: a streamed response lost its task and ended
//! without `[DONE]` or an error event (a cut stream), and a whole response
//! lost its connection without an answer; the runtime caught the panic and
//! nothing named the request. [`isolated`] runs one request's reading with
//! its panic caught at the request's own boundary, so the panic is that
//! request's failure ([`Panicked`]): the stream ends with the error event
//! under [`READER_PANICKED_CODE`] and `[DONE]`, the whole response is a 500
//! under the same code, the output is counted under the table that read it
//! (`smg_symphony_outputs_total{outcome="panicked"}`) and one ERROR line
//! names the request, the model, the table and the panic's message
//! ([`Panicked::record`]). Nothing else of the process is touched: the panic
//! hook prints its line as before, the other requests and the engine
//! streams keep going.
//!
//! The message in the line is the panic's own, sanitized: a slice panic
//! quotes the bytes it indexed (`byte index 7 is out of bounds of `...``),
//! which are model output and may carry the user's data, so every span
//! between backticks is elided and the message is cut to
//! [`MESSAGE_SHOWN`] bytes. The sanitization is best effort: it knows the
//! shapes the standard library's panics have, and the line is the
//! operator's log, never the client's. The text a client reads, on the HTTP
//! error body and on the SSE error event alike, says only that the
//! gateway's reader of the response panicked, under the code; nothing of
//! the message reaches it. The hook's line on stderr keeps the full text,
//! as it did.
//!
//! The cost in the common case: one landing pad per request (a whole
//! response) or per poll of the stream's task, no allocation; the test trap
//! ([`test_trap`]) is one branch per step of a reader, with the marker read
//! once per process.

use std::{any::Any, fmt, future::Future, panic::AssertUnwindSafe, sync::OnceLock};

use futures::FutureExt;
use tracing::error;

use crate::routers::grpc::regular::symphony::telemetry;

/// The client-visible code of a request whose reading panicked, on the HTTP
/// error body and on the SSE error event alike.
pub(crate) const READER_PANICKED_CODE: &str = "reader_panicked";

/// The bytes of a panic's message the log line carries at most.
const MESSAGE_SHOWN: usize = 256;

/// The environment variable that arms the test trap: the text a response
/// carries that makes the reader panic, for the tests that drive a panic
/// through a running gateway (the end-to-end mock class). Read once per
/// process; unset in every other one.
pub(crate) const TEST_TRAP_VAR: &str = "SMG_TEST_READER_PANIC_ON";

/// A request whose reading panicked, caught at the request's boundary. Its
/// `Display` is the text a client reads: the failure named, without the
/// panic's message, which is the log line's ([`Panicked::record`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Panicked {
    /// The panic's message, sanitized ([`sanitized`]).
    message: String,
    /// The `table` label of the Symphony table that was reading the
    /// response, when one was.
    table: Option<&'static str>,
}

impl Panicked {
    /// The panic's payload as the request's failure, under no table yet.
    fn from_payload(payload: &(dyn Any + Send)) -> Self {
        Self {
            message: sanitized(&panic_message(payload)),
            table: None,
        }
    }

    /// The same failure, read by the Symphony `table` (its label) when one
    /// read the response: the outcome is counted under it.
    pub(crate) fn read_by(mut self, table: Option<&'static str>) -> Self {
        self.table = table;
        self
    }

    /// The panic's message, sanitized.
    #[cfg(test)]
    pub(crate) fn message(&self) -> &str {
        &self.message
    }

    /// Records the failure: one increment under the table's `panicked`
    /// outcome when a table read the response, and one ERROR line with the
    /// request id, the model, the table and the message. Once per request,
    /// where the request's answer is decided.
    pub(crate) fn record(&self, request_id: &str, model: &str) {
        if let Some(table) = self.table {
            telemetry::record_panicked(table);
        }
        error!(
            request_id,
            model,
            table = self.table.unwrap_or("none"),
            panic.message = %self.message,
            "The response's reader panicked; the request ends with an error"
        );
    }
}

impl fmt::Display for Panicked {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the gateway's reader of the response panicked")
    }
}

/// Runs `work`, one request's reading, with a panic in it caught as the
/// request's failure: `Ok` with the work's output, `Err` with the panic.
/// A poll that panics ends the work; its state is dropped as the unwind
/// passes it, which is what a request's state is for.
pub(crate) async fn isolated<F: Future>(work: F) -> Result<F::Output, Panicked> {
    AssertUnwindSafe(work)
        .catch_unwind()
        .await
        .map_err(|payload| Panicked::from_payload(payload.as_ref()))
}

/// [`isolated`] for a reading that does not await: `work` run with its
/// panic caught.
pub(crate) fn isolated_sync<T>(work: impl FnOnce() -> T) -> Result<T, Panicked> {
    std::panic::catch_unwind(AssertUnwindSafe(work))
        .map_err(|payload| Panicked::from_payload(payload.as_ref()))
}

/// The text of a panic's payload: the `&str` or `String` a panic carries,
/// or a word for a payload of another type.
pub(crate) fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// `message` with every span between backticks elided (the bytes a slice
/// panic quotes are model output), an opening backtick that never closes
/// cutting the rest, and the whole cut to [`MESSAGE_SHOWN`] bytes on a
/// character boundary.
fn sanitized(message: &str) -> String {
    let mut out = String::with_capacity(message.len().min(MESSAGE_SHOWN));
    let mut rest = message;
    while let Some(open) = rest.find('`') {
        out.push_str(&rest[..open]);
        out.push_str("`...`");
        match rest[open + 1..].find('`') {
            Some(close) => rest = &rest[open + 1 + close + 1..],
            None => rest = "",
        }
    }
    out.push_str(rest);
    if out.len() > MESSAGE_SHOWN {
        let mut end = MESSAGE_SHOWN;
        while !out.is_char_boundary(end) {
            end -= 1;
        }
        out.truncate(end);
        out.push_str("...");
    }
    out
}

/// The text the test trap springs on, when this process runs with one
/// ([`TEST_TRAP_VAR`]; a test of this crate arms it with
/// [`arm_test_trap`]): a reader whose step carries it panics, so the
/// request's isolation is exercised end to end. `None` otherwise.
pub(crate) fn test_trap() -> Option<&'static str> {
    static TRAP: OnceLock<Option<Box<str>>> = OnceLock::new();
    #[cfg(test)]
    if let Some(armed) = ARMED.get() {
        return Some(armed);
    }
    TRAP.get_or_init(|| {
        std::env::var(TEST_TRAP_VAR)
            .ok()
            .filter(|marker| !marker.is_empty())
            .map(String::into_boxed_str)
    })
    .as_deref()
}

/// The marker the tests of this crate armed the trap with, once per test
/// process (the tests share it; a marker no other output carries).
#[cfg(test)]
static ARMED: OnceLock<&'static str> = OnceLock::new();

/// Arms the test trap with `marker` for the rest of the test process; a
/// marker armed earlier stays.
#[cfg(test)]
pub(crate) fn arm_test_trap(marker: &'static str) -> &'static str {
    ARMED.get_or_init(|| marker)
}

/// The test trap springing: a reader's step carried the marker. Cold, and
/// the one panic of the request path that exists in order to panic.
#[cold]
#[inline(never)]
#[expect(
    clippy::panic,
    reason = "the test trap exists to panic: it drives the request's isolation in tests"
)]
pub(crate) fn spring_test_trap(marker: &str) -> ! {
    // The marker is quoted the way a slice panic quotes its bytes, so the
    // sanitized message elides it as it would the bytes.
    panic!("the test trap sprang: the response carried `{marker}`");
}

#[cfg(test)]
mod tests {
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
}
