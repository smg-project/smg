//! Validate terminal events on HTTP PD Chat/Completions streams. HTTP EOF
//! alone says nothing about whether generation completed. Read complete SSE
//! frames before forwarding so an engine abort cannot become a successful
//! completion. The shared decoder bounds held state to one incomplete frame.

use bytes::{Bytes, BytesMut};
use serde_json::Value;

use super::{ChunkHook, ChunkStep};
use crate::routers::common::sse::SseDecoder;

pub(crate) struct CompletionStreamHook {
    decoder: SseDecoder,
    pending: BytesMut,
    next: Box<dyn ChunkHook>,
    terminal: bool,
}

impl CompletionStreamHook {
    pub(crate) fn new(next: Box<dyn ChunkHook>) -> Self {
        Self {
            decoder: SseDecoder::new(),
            pending: BytesMut::new(),
            next,
            terminal: false,
        }
    }

    fn failed(output: Vec<u8>, reason: &'static str) -> ChunkStep {
        ChunkStep {
            bytes: Bytes::from(output),
            last: true,
            failure: Some(reason),
        }
    }
}

impl ChunkHook for CompletionStreamHook {
    fn read(&mut self, bytes: Bytes) -> ChunkStep {
        let mut output = Vec::with_capacity(bytes.len());
        // Drain between bounded pushes: a large HTTP chunk containing many
        // small frames must not overflow the decoder's single-frame budget.
        for part in bytes.chunks(16 * 1024) {
            if self.decoder.push(part).is_err() {
                return Self::failed(output, "decode SSE frame exceeded the buffer limit");
            }
            self.pending.extend_from_slice(part);
            loop {
                let frame = self.decoder.next_frame();
                let consumed = self.pending.len() - self.decoder.buffered_len();
                let raw = self.pending.split_to(consumed).freeze();
                match frame {
                    Some(Err(_)) => {
                        return Self::failed(output, "decode sent an invalid UTF-8 SSE frame");
                    }
                    Some(Ok(frame)) => {
                        if frame.is_done() {
                            self.terminal = true;
                        } else if let Ok(value) = frame.decode_data::<Value>() {
                            if value.get("error").is_some_and(|error| !error.is_null()) {
                                return Self::failed(output, "decode reported a stream error");
                            }
                            let aborted = value
                                .get("choices")
                                .and_then(Value::as_array)
                                .is_some_and(|choices| {
                                    choices.iter().any(|choice| {
                                        matches!(
                                            choice.get("finish_reason").and_then(Value::as_str),
                                            Some("abort" | "error")
                                        )
                                    })
                                });
                            if aborted {
                                return Self::failed(
                                    output,
                                    "decode aborted generation before completion",
                                );
                            }
                        }
                        let step = self.next.read(raw);
                        output.extend_from_slice(&step.bytes);
                        if step.failure.is_some() || step.last || self.terminal {
                            return ChunkStep {
                                bytes: Bytes::from(output),
                                last: true,
                                failure: step.failure,
                            };
                        }
                    }
                    None => {
                        // The decoder consumes comment/control-only blocks
                        // too; preserve their original bytes and keepalives.
                        if !raw.is_empty() {
                            let step = self.next.read(raw);
                            output.extend_from_slice(&step.bytes);
                            if step.failure.is_some() || step.last {
                                return ChunkStep {
                                    bytes: Bytes::from(output),
                                    last: step.last,
                                    failure: step.failure,
                                };
                            }
                        }
                        break;
                    }
                }
            }
        }
        ChunkStep {
            bytes: Bytes::from(output),
            last: false,
            failure: None,
        }
    }

    fn finish(&mut self) -> ChunkStep {
        // A held frame was never complete. Forwarding it before an error
        // separator could turn partial DONE/abort into a valid client event.
        // This hook owns those unforwarded bytes; quarantine them on EOF and
        // transport failure alike instead of completing them for the worker.
        self.pending.clear();
        let tail = self.next.finish();
        ChunkStep {
            bytes: tail.bytes,
            last: true,
            failure: if self.terminal {
                tail.failure
            } else {
                Some("decode stream ended before [DONE]")
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hook() -> CompletionStreamHook {
        CompletionStreamHook::new(Box::new(|bytes: Bytes| ChunkStep {
            bytes,
            last: false,
            failure: None,
        }))
    }

    fn through(input: &[u8], width: usize) -> (Vec<u8>, Option<&'static str>, bool) {
        let mut hook = hook();
        let mut output = Vec::new();
        for part in input.chunks(width) {
            let step = hook.read(Bytes::copy_from_slice(part));
            output.extend_from_slice(&step.bytes);
            if step.failure.is_some() || step.last {
                return (output, step.failure, step.last);
            }
        }
        let tail = hook.finish();
        output.extend_from_slice(&tail.bytes);
        (output, tail.failure, false)
    }

    #[test]
    fn normal_choice_reasons_and_usage_survive_every_chunk_split() {
        for reason in [
            "stop",
            "length",
            "tool_calls",
            "function_call",
            "content_filter",
        ] {
            let input = format!("data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"café\"}},\"finish_reason\":\"{reason}\"}}]}}\r\n\r\ndata: {{\"choices\":[],\"usage\":{{\"completion_tokens\":1}}}}\r\n\r\ndata: [DONE]\r\n\r\n");
            for width in 1..=input.len() {
                let (output, error, terminal) = through(input.as_bytes(), width);
                // A CR at an HTTP chunk boundary is a complete SSE EOL.
                // A final LF may be after the terminal event and need not
                // keep the upstream alive; all payload bytes stay exact.
                assert_eq!(
                    output.trim_ascii_end(),
                    input.as_bytes().trim_ascii_end(),
                    "{reason} split {width}"
                );
                assert!(error.is_none(), "{reason} split {width}: {error:?}");
                assert!(terminal);
            }
        }
    }

    #[test]
    fn multiple_choices_are_not_ended_by_the_first_finish_or_usage_tail() {
        let input = b"data: {\"choices\":[{\"index\":0,\"finish_reason\":\"stop\"},{\"index\":1,\"finish_reason\":null}]}\n\ndata: {\"choices\":[{\"index\":1,\"finish_reason\":\"length\"}]}\n\ndata: {\"choices\":[],\"usage\":{}}\n\ndata: [DONE]\n\n";
        let (output, error, terminal) = through(input, 1);
        assert_eq!(output, input);
        assert!(error.is_none());
        assert!(terminal);
    }

    #[test]
    fn abort_never_reaches_the_client_as_a_completion() {
        let input = b"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\ndata: {\"choices\":[{\"index\":0,\"finish_reason\":\"abort\"}]}\n\ndata: [DONE]\n\n";
        for width in 1..=input.len() {
            let (output, error, _) = through(input, width);
            assert!(error.is_some(), "split {width}");
            assert!(!String::from_utf8_lossy(&output).contains("abort"));
            assert!(String::from_utf8_lossy(&output).contains("content"));
        }
    }

    #[test]
    fn backend_error_and_error_finish_reason_fail() {
        for input in [
            &b"data: {\"error\":{\"message\":\"engine died\"}}\n\n"[..],
            &b"data: {\"choices\":[{\"finish_reason\":\"error\"}]}\n\n"[..],
        ] {
            let (output, error, _) = through(input, 2);
            assert!(output.is_empty());
            assert!(error.is_some());
        }
    }

    #[test]
    fn fake_done_and_metadata_do_not_hide_eof() {
        let input = b": keepalive\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"[DONE] abort\"}}]}\n\ndata: {\"choices\":[],\"usage\":{}}\n\n";
        let (output, error, terminal) = through(input, 3);
        assert_eq!(output, input);
        assert_eq!(error, Some("decode stream ended before [DONE]"));
        assert!(!terminal);
    }

    #[test]
    fn incomplete_sentinel_or_json_cannot_finish_a_stream() {
        for input in [&b"data: [DONE]\n"[..], &b"data: {\"choices\":["[..]] {
            let (output, error, terminal) = through(input, 1);
            assert!(
                output.is_empty(),
                "unterminated data must not be promoted by error framing"
            );
            assert!(error.is_some());
            assert!(!terminal);
        }
    }

    #[test]
    fn no_bytes_after_done_are_relayed() {
        let input = b"data: [DONE]\n\ndata: after\n\n";
        let (output, error, terminal) = through(input, input.len());
        assert_eq!(output, b"data: [DONE]\n\n");
        assert!(error.is_none());
        assert!(terminal);
    }

    #[test]
    fn an_unbounded_unterminated_frame_is_rejected() {
        let input = vec![b'x'; 1024 * 1024 + 1];
        let (output, error, _) = through(&input, 16 * 1024);
        assert!(output.is_empty());
        assert!(error.is_some());
    }
}
