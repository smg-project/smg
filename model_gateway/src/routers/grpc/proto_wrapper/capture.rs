//! Opt-in observation of decoded native vLLM gRPC messages, before router transformations.
//!
//! This preserves the known protobuf fields at this SMG revision, not original network bytes
//! or unknown fields discarded by Prost. Status code, message and details are retained; status
//! metadata is not. The protocol does not attest the worker's runtime.
//! No serving path installs this automatically. Its caller must bind a fresh stream and unique
//! source ID to the request/configuration, and separately validate semantic completeness.

use base64::{display::Base64Display, engine::general_purpose::STANDARD};
use futures_util::StreamExt;
use prost::Message;
use serde::{ser::Error as _, Serialize, Serializer};

use super::{vllm, VllmStream};
use crate::diagnostic_capture::CaptureObserver;

/// One explicitly observed native transport, retaining its normal abort-on-drop behavior.
///
/// The observation ends when this wrapper is dropped. A native deferred-abort cleanup may
/// drain further messages afterward; those are not reader-visible ingress and are not recorded.
/// Drop records the observed prefix's closure, not successful inference or complete capture.
pub struct CapturedVllmStream {
    stream: VllmStream,
    observation: Observation,
}

impl CapturedVllmStream {
    pub(super) fn new(stream: VllmStream, capture: CaptureObserver, source: u64) -> Self {
        let mut observation = Observation {
            capture,
            source,
            ordinal: 0,
            items: 0,
            eof: false,
            completion_marked: false,
        };
        observation.record(Event::Open {});
        Self {
            stream,
            observation,
        }
    }

    /// Observe each yielded native result before the router maps engine errors or restamps it.
    pub(super) async fn next(&mut self) -> Option<Result<vllm::GenerateResponse, tonic::Status>> {
        let item = self.stream.next().await;
        let limit = self.observation.capture.record_bytes_limit();
        match &item {
            Some(result) => {
                self.observation.count_item();
                match result {
                    Ok(message) => self.observation.record(Event::Message {
                        protobuf_base64: Protobuf { message, limit },
                    }),
                    Err(status) => self.observation.record(Event::Status {
                        code: status.code() as i32,
                        message: LimitedText {
                            text: status.message(),
                            limit,
                        },
                        details_base64: Binary {
                            bytes: status.details(),
                            limit,
                        },
                    }),
                }
            }
            None if !self.observation.eof => {
                self.observation.eof = true;
                self.observation.record(Event::Eof {});
            }
            None => {}
        }
        // A diagnostic failure is latched in the completion owner, never substituted for an
        // inference result. The source's original response/status always reaches the router.
        item
    }

    pub(super) fn mark_completed(&mut self) {
        self.stream.mark_completed();
        self.observation.completion_marked = true;
    }

    pub(super) fn defer_abort_until_first_item(self) -> Self {
        let Self {
            stream,
            observation,
        } = self;
        Self {
            stream: stream.defer_abort_until_first_item(),
            observation,
        }
    }
}

struct Observation {
    capture: CaptureObserver,
    source: u64,
    ordinal: u64,
    items: u64,
    eof: bool,
    completion_marked: bool,
}

impl Observation {
    fn count_item(&mut self) {
        if let Some(next) = self.items.checked_add(1) {
            self.items = next;
        } else {
            self.capture.invalidate();
        }
    }

    fn record(&mut self, event: Event<'_>) {
        let Some(next) = self.ordinal.checked_add(1) else {
            self.capture.invalidate();
            return;
        };
        // record() latches any failure; the owner must observe finish(). The diagnostic path
        // deliberately does not replace successful inference or mask its original error.
        let _ = self.capture.record(&Envelope {
            schema: "smg.vllm_grpc_ingress.v1",
            source: self.source,
            ordinal: self.ordinal,
            event,
        });
        self.ordinal = next;
    }
}

impl Drop for Observation {
    fn drop(&mut self) {
        self.record(Event::Closed {
            items: self.items,
            eof: self.eof,
            completion_marked: self.completion_marked,
        });
    }
}

#[derive(Serialize)]
struct Envelope<'a> {
    schema: &'static str,
    source: u64,
    ordinal: u64,
    #[serde(flatten)]
    event: Event<'a>,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Event<'a> {
    Open {},
    Message {
        protobuf_base64: Protobuf<'a>,
    },
    Status {
        code: i32,
        message: LimitedText<'a>,
        details_base64: Binary<'a>,
    },
    Eof {},
    Closed {
        items: u64,
        eof: bool,
        completion_marked: bool,
    },
}

struct Protobuf<'a> {
    message: &'a vllm::GenerateResponse,
    limit: usize,
}

impl Serialize for Protobuf<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // This runs only after the shared observer and queue admit the serializer. Reject the
        // source length before allocating an encoding buffer; base64 expansion is included.
        // The capped JSON writer still enforces envelope overhead and final serialized bytes.
        let len = self.message.encoded_len();
        check_binary_len::<S::Error>(len, self.limit)?;
        let mut encoded = Vec::with_capacity(len);
        self.message
            .encode(&mut encoded)
            .map_err(S::Error::custom)?;
        serializer.collect_str(&Base64Display::new(&encoded, &STANDARD))
    }
}

struct Binary<'a> {
    bytes: &'a [u8],
    limit: usize,
}

impl Serialize for Binary<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        check_binary_len::<S::Error>(self.bytes.len(), self.limit)?;
        serializer.collect_str(&Base64Display::new(self.bytes, &STANDARD))
    }
}

fn check_binary_len<E: serde::ser::Error>(bytes: usize, limit: usize) -> Result<(), E> {
    if base64::encoded_len(bytes, true).is_none_or(|len| len > limit) {
        return Err(E::custom("native capture record exceeds its byte limit"));
    }
    Ok(())
}

struct LimitedText<'a> {
    text: &'a str,
    limit: usize,
}

impl Serialize for LimitedText<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if self.text.len() > self.limit {
            return Err(S::Error::custom(
                "native capture text exceeds its byte limit",
            ));
        }
        serializer.serialize_str(self.text)
    }
}

#[cfg(test)]
#[path = "capture_tests.rs"]
mod tests;
