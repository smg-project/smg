use std::pin::Pin;

use bytes::Bytes;
use futures::stream::Stream;
use tokio_stream::StreamExt;

/// A parsed SSE event.
#[derive(Debug, Clone)]
pub struct SseEvent {
    /// The `event:` field (e.g. "message_start" for Anthropic).
    pub event: Option<String>,
    /// The `data:` field (typically JSON).
    pub data: String,
}

/// Transform a byte stream from reqwest into a stream of [`SseEvent`]s.
///
/// Handles the three SSE protocol variants:
/// - OpenAI: `data: {...}\n\n` terminated by `data: [DONE]`
/// - Anthropic: `event: type\ndata: {...}\n\n`
/// - Responses API: `event: type\ndata: {...}\n\n` + `data: [DONE]`
pub(crate) fn sse_stream(
    byte_stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + Unpin + 'static,
) -> Pin<Box<dyn Stream<Item = Result<SseEvent, crate::SmgError>> + Send>> {
    let state = SseParserState::default();
    // Holds trailing bytes from an incomplete UTF-8 sequence split across chunks.
    let incomplete: Vec<u8> = Vec::new();
    Box::pin(futures::stream::unfold(
        (byte_stream, state, String::new(), incomplete),
        |(mut stream, mut state, mut buffer, mut incomplete)| async move {
            loop {
                // Try to parse a complete event from the buffer.
                if let Some(event) = state.try_parse(&mut buffer) {
                    return Some((Ok(event), (stream, state, buffer, incomplete)));
                }

                // [DONE] was received — terminate the stream.
                if state.done {
                    return None;
                }

                // Need more data from the stream.
                match stream.next().await {
                    Some(Ok(chunk)) => {
                        // Prepend any leftover bytes from a split UTF-8 sequence.
                        let bytes = if incomplete.is_empty() {
                            chunk.to_vec()
                        } else {
                            let mut combined = std::mem::take(&mut incomplete);
                            combined.extend_from_slice(&chunk);
                            combined
                        };
                        match String::from_utf8(bytes) {
                            Ok(s) => buffer.push_str(&s),
                            Err(e) => {
                                let valid_up_to = e.utf8_error().valid_up_to();
                                let raw = e.into_bytes();
                                // SAFETY: valid_up_to is guaranteed valid UTF-8.
                                buffer.push_str(
                                    std::str::from_utf8(&raw[..valid_up_to]).unwrap_or_default(),
                                );
                                // Stash the trailing incomplete sequence for the next chunk.
                                incomplete = raw[valid_up_to..].to_vec();
                            }
                        }
                    }
                    Some(Err(e)) => {
                        return Some((
                            Err(crate::SmgError::Connection(e)),
                            (stream, state, buffer, incomplete),
                        ));
                    }
                    None => {
                        // Stream ended. Try to flush any remaining event.
                        if let Some(event) = state.flush(&mut buffer) {
                            return Some((Ok(event), (stream, state, buffer, incomplete)));
                        }
                        return None;
                    }
                }
            }
        },
    ))
}

#[derive(Default)]
struct SseParserState {
    current_event: Option<String>,
    current_data: Vec<String>,
    /// Set when `data: [DONE]` is received, signaling the stream should end.
    done: bool,
}

impl SseParserState {
    /// Try to parse a complete SSE event from the buffer.
    /// Returns `Some(event)` if a complete event was found and consumed.
    fn try_parse(&mut self, buffer: &mut String) -> Option<SseEvent> {
        loop {
            // Find the next line boundary.
            let newline_pos = buffer.find('\n')?;
            let line = buffer[..newline_pos].trim_end_matches('\r').to_string();
            // Consume the line + newline from buffer.
            buffer.drain(..=newline_pos);

            if line.is_empty() {
                // Empty line = event boundary.
                if !self.current_data.is_empty() {
                    let data = self.current_data.join("\n");
                    let event_type = self.current_event.take();
                    self.current_data.clear();

                    // `[DONE]` signals the end of the stream (OpenAI/Responses API).
                    if data == "[DONE]" {
                        self.done = true;
                        return None;
                    }

                    return Some(SseEvent {
                        event: event_type,
                        data,
                    });
                }
                // Empty line with no data — discard any stale event type
                // so it doesn't leak into the next event.
                self.current_event = None;
                continue;
            }

            if let Some(value) = line
                .strip_prefix("data: ")
                .or_else(|| line.strip_prefix("data:"))
            {
                self.current_data.push(value.to_string());
            } else if let Some(value) = line
                .strip_prefix("event: ")
                .or_else(|| line.strip_prefix("event:"))
            {
                self.current_event = Some(value.to_string());
            }
            // Ignore other fields (id:, retry:, comments starting with :).
        }
    }

    /// Flush any remaining event data when the stream ends.
    /// Processes any partial line left in the buffer before assembling the event.
    fn flush(&mut self, buffer: &mut String) -> Option<SseEvent> {
        // Process any remaining partial line in the buffer.
        if !buffer.is_empty() {
            let line = buffer.trim_end_matches(['\r', '\n']);
            if let Some(value) = line
                .strip_prefix("data: ")
                .or_else(|| line.strip_prefix("data:"))
            {
                self.current_data.push(value.to_string());
            } else if let Some(value) = line
                .strip_prefix("event: ")
                .or_else(|| line.strip_prefix("event:"))
            {
                self.current_event = Some(value.to_string());
            }
            buffer.clear();
        }

        if !self.current_data.is_empty() {
            let data = self.current_data.join("\n");
            let event_type = self.current_event.take();
            self.current_data.clear();

            if data == "[DONE]" {
                return None;
            }

            return Some(SseEvent {
                event: event_type,
                data,
            });
        }
        None
    }
}

#[cfg(test)]
#[path = "../../tests/unit/streaming/sse/tests.rs"]
mod tests;
