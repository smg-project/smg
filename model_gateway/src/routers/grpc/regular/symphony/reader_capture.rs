//! Explicit Reader observations, borrowed directly into the admitted bounded writer.
//!
//! Operations preserve the distinction between decoding tokens, releasing decoder tail and
//! ending the reader. Inputs alone replay Engine, not Reader's token accounting or held text.
//! Complete batches are recorded at the streaming Chat adapter entrance, before consumption.
//! This transcript has no source/tokenizer/format-selector attestation and grants no replay
//! eligibility. The request coordinator must bind those and check terminal boundary coverage.

use serde::Serialize;

use super::{Assignment, CallIds, EngineFinish, Event, Events, Input, Tool};
use crate::diagnostic_capture::{CaptureObserver, FormatRecipe};

pub(super) struct Observation {
    capture: CaptureObserver,
    source: u64,
    choice: u32,
    ordinal: u64,
    inputs: u64,
    batches: u64,
    pending: Option<OperationKind>,
    finished: bool,
    failed: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum OperationKind {
    Construct,
    Read,
    Tail,
    End,
}

#[derive(Serialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub(super) enum Operation<'a> {
    Read { token_ids: &'a [u32] },
    Tail { text: &'a str },
    End { finish: &'a EngineFinish },
}

impl Operation<'_> {
    fn kind(&self) -> OperationKind {
        match self {
            Self::Read { .. } => OperationKind::Read,
            Self::Tail { .. } => OperationKind::Tail,
            Self::End { .. } => OperationKind::End,
        }
    }
}

impl Observation {
    pub(super) fn new(
        capture: CaptureObserver,
        source: u64,
        choice: u32,
        assignment: &Assignment,
        tools: &[Tool],
        ids: CallIds,
    ) -> Self {
        let mut observation = Self {
            capture,
            source,
            choice,
            ordinal: 0,
            inputs: 0,
            batches: 0,
            pending: Some(OperationKind::Construct),
            finished: false,
            failed: false,
        };
        observation.record(Record::Open {
            format_name: assignment.format.name(),
            recipe: &assignment.recipe,
            tools,
            nonce: &ids.nonce(),
            first_call_only: false,
        });
        observation
    }

    pub(super) fn begin(&mut self, operation: Operation<'_>) {
        if self.pending.is_some() {
            // Continuing a reader whose prior operation unwound cannot silently erase the
            // missing return. Keep inference unchanged, but reject completion of its evidence.
            self.reject("previous reader operation did not return", None);
        }
        self.pending = Some(operation.kind());
        self.record(Record::Operation { step: operation });
    }

    pub(super) fn returned(&mut self, stopped: Option<bool>) {
        let Some(operation) = self.pending.take() else {
            self.reject("reader returned without a pending operation", None);
            return;
        };
        if operation == OperationKind::End {
            self.finished = true;
        }
        self.record(Record::Returned { operation, stopped });
    }

    pub(super) fn policy(&mut self, first_call_only: bool) {
        self.record(Record::Policy { first_call_only });
    }

    pub(super) fn input(&mut self, input: &Input<'_>) {
        if increment(&mut self.inputs).is_none() {
            self.reject("reader input counter overflow", None);
        }
        self.record(Record::Input { input });
    }

    pub(super) fn failed(&mut self, message: &str) {
        self.failed = true;
        self.record(Record::Failed { message });
    }

    pub(super) fn batch(&mut self, choice: u32, events: &Events<'_>) {
        if choice != self.choice {
            self.reject("chat batch choice differs from reader choice", Some(choice));
            return;
        }
        if increment(&mut self.batches).is_none() {
            self.reject("reader batch counter overflow", None);
            return;
        }
        self.record(Record::ChatBatch {
            events: events.as_slice(),
        });
    }

    fn reject(&mut self, reason: &'static str, attempted_choice: Option<u32>) {
        // Preserve the cause when the sink still accepts records. Latching first would make
        // the otherwise valid prefix indistinguishable from an unexplained observer failure.
        self.record(Record::Rejected {
            reason,
            attempted_choice,
            pending: self.pending,
        });
        self.capture.invalidate();
    }

    fn record(&mut self, record: Record<'_>) {
        let Some(ordinal) = increment(&mut self.ordinal) else {
            self.capture.invalidate();
            return;
        };
        // SharedCapture latches serialization/queue failures. The owner must check finish();
        // diagnostics must never replace inference output or its original error.
        let _ = self.capture.record(&Envelope {
            schema: "smg.symphony_reader.v2",
            source: self.source,
            choice: self.choice,
            ordinal,
            record,
        });
    }
}

impl Drop for Observation {
    fn drop(&mut self) {
        self.record(Record::Closed {
            inputs: self.inputs,
            batches: self.batches,
            finished: self.finished,
            failed: self.failed,
            pending: self.pending,
            unwinding: std::thread::panicking(),
        });
    }
}

/// Return the previous value, reserving the next ordinal without wrapping.
fn increment(value: &mut u64) -> Option<u64> {
    let previous = *value;
    *value = value.checked_add(1)?;
    Some(previous)
}

#[derive(Serialize)]
struct Envelope<'a> {
    schema: &'static str,
    source: u64,
    choice: u32,
    ordinal: u64,
    #[serde(flatten)]
    record: Record<'a>,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Record<'a> {
    Open {
        format_name: &'a str,
        recipe: &'a Option<FormatRecipe>,
        tools: &'a [Tool],
        nonce: &'a str,
        first_call_only: bool,
    },
    Policy {
        first_call_only: bool,
    },
    Operation {
        #[serde(flatten)]
        step: Operation<'a>,
    },
    Returned {
        operation: OperationKind,
        stopped: Option<bool>,
    },
    Input {
        input: &'a Input<'a>,
    },
    Failed {
        message: &'a str,
    },
    Rejected {
        reason: &'static str,
        attempted_choice: Option<u32>,
        pending: Option<OperationKind>,
    },
    ChatBatch {
        events: &'a [Event<'a>],
    },
    Closed {
        inputs: u64,
        batches: u64,
        finished: bool,
        failed: bool,
        pending: Option<OperationKind>,
        // True only when this handle is dropped during unwind, not a claim about a panic caught
        // while the reader stayed alive. A missing operation return is retained in `pending`.
        unwinding: bool,
    },
}

#[cfg(test)]
#[path = "reader_capture_tests.rs"]
mod tests;
