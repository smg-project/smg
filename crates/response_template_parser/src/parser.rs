use std::{collections::BTreeSet, sync::Arc};

use serde_json::{Map, Value};

use crate::{
    compiled::{CompiledField, CompiledTemplate, ContentKind},
    output::{ParseOutput, ToolCall},
    DelimiterBound, ParserConfig, ResponseTemplate, ResponseTemplateError,
};

/// A validated, reusable response-template parser.
#[derive(Debug, Clone)]
pub struct ResponseTemplateParser {
    compiled: Arc<CompiledTemplate>,
    config: ParserConfig,
}

/// Whether the backend exhausted its generation budget or ended normally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishMode {
    /// Require every opened field to have its declared closing delimiter.
    Strict,
    /// Accept an unfinished text field at a confirmed generation length limit.
    /// Structured fields, unknown bytes, UTF-8 and size limits remain strict.
    LengthLimit,
}

impl ResponseTemplateParser {
    /// Validate and compile `template`. `model_name` labels every error.
    pub fn new(
        model_name: impl Into<String>,
        template: ResponseTemplate,
        config: ParserConfig,
    ) -> Result<Self, ResponseTemplateError> {
        let compiled = CompiledTemplate::new(model_name.into(), template, config)?;
        Ok(Self {
            compiled: Arc::new(compiled),
            config,
        })
    }

    /// Decode a raw `response_template` value, then validate and compile it.
    /// Unknown keys are rejected.
    pub fn from_json(
        model_name: impl Into<String>,
        template: &Value,
        config: ParserConfig,
    ) -> Result<Self, ResponseTemplateError> {
        let model_name = model_name.into();
        let decoded = serde_json::from_value(template.clone()).map_err(|error| {
            ResponseTemplateError::InvalidTemplate {
                model_name: model_name.clone(),
                field: "response_template".to_string(),
                limit: 0,
                reason: error.to_string(),
            }
        })?;
        Self::new(model_name, decoded, config)
    }

    /// Width bound of each compiled delimiter pattern.
    pub fn delimiter_metadata(&self) -> &[crate::DelimiterMetadata] {
        &self.compiled.metadata
    }

    /// Every literal close delimiter declared by the template's fields.
    pub fn close_literals(&self) -> impl Iterator<Item = &str> {
        self.compiled
            .fields
            .iter()
            .flat_map(|field| field.closes.iter().map(String::as_str))
    }

    /// Start independent streaming state for one response. The rendered
    /// prompt prefix must contain the template's start anchor.
    pub fn stream(&self, rendered_prompt_prefix: impl Into<String>) -> StreamingParser {
        StreamingParser {
            compiled: Arc::clone(&self.compiled),
            config: self.config,
            rendered_prompt_prefix: rendered_prompt_prefix.into(),
            anchor_validated: false,
            bytes: Vec::new(),
            incomplete_utf8: 0,
            open_field: None,
            seen_fields: BTreeSet::new(),
            poison: None,
            finished: false,
        }
    }

    /// Parse one complete response with strict finalization.
    pub fn parse_complete(
        &self,
        rendered_prompt_prefix: &str,
        decoded_output: &str,
    ) -> Result<ParseOutput, ResponseTemplateError> {
        self.parse_complete_with_mode(rendered_prompt_prefix, decoded_output, FinishMode::Strict)
    }

    /// Parse one complete response, finalizing it with `mode`.
    pub fn parse_complete_with_mode(
        &self,
        rendered_prompt_prefix: &str,
        decoded_output: &str,
        mode: FinishMode,
    ) -> Result<ParseOutput, ResponseTemplateError> {
        let mut stream = self.stream(rendered_prompt_prefix);
        let mut output = stream.feed(decoded_output.as_bytes())?;
        output.merge(stream.finish_with_mode(mode)?);
        Ok(output)
    }
}

/// Per-request streaming state. Input is accepted as bytes so a caller may
/// split in the middle of a UTF-8 scalar. Completed fields are emitted and
/// drained; only the bounded undecidable suffix remains in `bytes`.
#[derive(Debug)]
pub struct StreamingParser {
    compiled: Arc<CompiledTemplate>,
    config: ParserConfig,
    rendered_prompt_prefix: String,
    anchor_validated: bool,
    bytes: Vec<u8>,
    /// Length of the incomplete UTF-8 scalar at the end of `bytes`.
    incomplete_utf8: usize,
    /// Set while `bytes` holds only an unfinished field.
    open_field: Option<OpenField>,
    seen_fields: BTreeSet<String>,
    poison: Option<ResponseTemplateError>,
    finished: bool,
}

/// An unfinished field that starts the retained buffer. Its body can only
/// end at a close literal, so later feeds search just the new bytes.
#[derive(Debug, Clone, Copy)]
struct OpenField {
    /// Index into `CompiledTemplate::fields`.
    field: usize,
    /// Ignorable bytes retained before the field opener.
    retained_prefix: usize,
    /// Offset in `bytes` where the field body starts.
    body_start: usize,
    /// No close literal starts before this offset in `bytes`.
    close_search_from: usize,
}

impl StreamingParser {
    /// Accept the next output bytes and return every field completed by them.
    /// An error poisons the stream: later calls return the same error.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<ParseOutput, ResponseTemplateError> {
        if let Some(error) = &self.poison {
            return Err(error.clone());
        }
        if self.finished {
            return Err(self.poison(ResponseTemplateError::RuntimeFailure {
                model_name: self.compiled.model_name.clone(),
                field: "stream".to_string(),
                limit: 0,
                reason: "feed called after finish".to_string(),
            }));
        }
        self.validate_start_anchor()?;

        // Append in place; a failed feed restores the last committed buffer.
        let committed_len = self.bytes.len();
        self.bytes.extend_from_slice(chunk);
        let result = match self.open_field {
            Some(open) => self.feed_open_field(open, committed_len),
            None => self.feed_retained(),
        };
        result.map_err(|error| {
            self.bytes.truncate(committed_len);
            self.poison(error)
        })
    }

    /// Validate and search only the appended bytes of an unfinished field,
    /// plus enough earlier bytes to find a close literal split across feeds.
    /// A feed that completes the field, or that could exceed a limit, falls
    /// back to a full scan.
    fn feed_open_field(
        &mut self,
        open: OpenField,
        committed_len: usize,
    ) -> Result<ParseOutput, ResponseTemplateError> {
        let checked = committed_len - self.incomplete_utf8;
        let valid_end = utf8_valid_end(&self.compiled, &self.bytes, checked)?;
        let field = &self.compiled.fields[open.field];
        let search_start = floor_char_boundary(&self.bytes, open.close_search_from);
        record_feed_scan(self.bytes.len() - search_start);
        let window = utf8_str(&self.compiled, &self.bytes, search_start, valid_end)?;
        if earliest_close(window, &field.closes).is_some() {
            return self.feed_retained();
        }
        // The window holds at least the last close length - 1 body bytes, so
        // the held close prefix matches a scan of the whole body.
        let held = longest_close_prefix_suffix(window, &field.closes);
        let body = valid_end - open.body_start - held;
        let incomplete = self.bytes.len() - valid_end;
        let within_limits = body <= self.config.max_body_bytes
            && match &field.tag {
                None => open.retained_prefix + held + incomplete <= self.config.max_pending_bytes,
                Some(tag) => {
                    // Every tag, key and value that the full scan checks lies
                    // within the body, and its pending count is the retained
                    // prefix plus at most one tag width of the text after the
                    // opener. While these bounds hold, no tag check can fail,
                    // so the tag regexes wait for the close.
                    let tail = valid_end - open.body_start;
                    let widest_tag = match tag.bound {
                        DelimiterBound::Bounded(maximum) => tail.min(maximum),
                        DelimiterBound::Unbounded => tail,
                    };
                    body <= self.config.max_structured_field_bytes
                        && open.retained_prefix + widest_tag + incomplete
                            <= self.config.max_pending_bytes
                }
            };
        if !within_limits {
            // A limit may be exceeded: the full scan decides exactly.
            return self.feed_retained();
        }
        self.incomplete_utf8 = incomplete;
        self.open_field = Some(OpenField {
            close_search_from: close_search_from(field, open.body_start, valid_end),
            ..open
        });
        Ok(ParseOutput::default())
    }

    /// Parse every retained byte: until an unfinished field starts the buffer,
    /// when such a field closes, and when an open field nears a limit.
    fn feed_retained(&mut self) -> Result<ParseOutput, ResponseTemplateError> {
        record_feed_scan(self.bytes.len());
        let valid_end = utf8_valid_end(&self.compiled, &self.bytes, 0)?;
        let valid = utf8_str(&self.compiled, &self.bytes, 0, valid_end)?;
        validate_runtime_limits(&self.compiled, self.config, valid)?;
        let mut staged_seen = self.seen_fields.clone();
        let (output, consumed) =
            consume_available(&self.compiled, self.config, valid, None, &mut staged_seen)?;
        let (pending, open_field) = pending_state(&self.compiled, &valid[consumed..]);
        let incomplete = self.bytes.len() - valid_end;
        ensure_pending_limit(&self.compiled, self.config, pending + incomplete)?;
        self.bytes.drain(..consumed);
        self.seen_fields = staged_seen;
        self.incomplete_utf8 = incomplete;
        self.open_field = open_field;
        Ok(output)
    }

    /// Finalize strictly: every opened field must be closed. Applies defaults.
    pub fn finish(&mut self) -> Result<ParseOutput, ResponseTemplateError> {
        self.finish_with_mode(FinishMode::Strict)
    }

    /// Finalize the stream with `mode`. Applies defaults for omitted fields.
    pub fn finish_with_mode(
        &mut self,
        mode: FinishMode,
    ) -> Result<ParseOutput, ResponseTemplateError> {
        if let Some(error) = &self.poison {
            return Err(error.clone());
        }
        if self.finished {
            return Err(self.poison(ResponseTemplateError::RuntimeFailure {
                model_name: self.compiled.model_name.clone(),
                field: "stream".to_string(),
                limit: 0,
                reason: "finish called more than once".to_string(),
            }));
        }
        self.validate_start_anchor()?;
        let text = match std::str::from_utf8(&self.bytes) {
            Ok(text) => text,
            Err(error) => {
                return Err(self.poison(ResponseTemplateError::RuntimeFailure {
                    model_name: self.compiled.model_name.clone(),
                    field: "utf8".to_string(),
                    limit: 0,
                    reason: format!(
                        "incomplete or invalid UTF-8 at byte {}",
                        error.valid_up_to()
                    ),
                }));
            }
        };
        if let Err(error) = validate_runtime_limits(&self.compiled, self.config, text) {
            return Err(self.poison(error));
        }
        let mut staged_seen = self.seen_fields.clone();
        let (mut output, consumed) = match consume_available(
            &self.compiled,
            self.config,
            text,
            Some(mode),
            &mut staged_seen,
        ) {
            Ok(result) => result,
            Err(error) => return Err(self.poison(error)),
        };
        if consumed != text.len() {
            return Err(self.poison(runtime(
                &self.compiled,
                "response",
                self.config.max_pending_bytes,
                "unconsumed assistant output at finish",
            )));
        }
        self.bytes.clear();
        self.incomplete_utf8 = 0;
        self.open_field = None;
        self.seen_fields = staged_seen;
        apply_defaults(&self.compiled, &self.seen_fields, &mut output)?;
        self.finished = true;
        Ok(output)
    }

    /// Retained bytes that have not been emitted yet.
    pub fn pending_bytes(&self) -> usize {
        self.bytes.len()
    }

    fn validate_start_anchor(&mut self) -> Result<(), ResponseTemplateError> {
        if self.anchor_validated {
            return Ok(());
        }
        if self.rendered_prompt_prefix.is_empty()
            || !self
                .compiled
                .start_anchor
                .is_match(&self.rendered_prompt_prefix)
        {
            return Err(self.poison(ResponseTemplateError::RuntimeFailure {
                model_name: self.compiled.model_name.clone(),
                field: "start_anchor_pattern".to_string(),
                limit: self.config.max_pending_bytes,
                reason: "rendered prompt prefix does not contain the declared start anchor"
                    .to_string(),
            }));
        }
        self.anchor_validated = true;
        Ok(())
    }

    fn poison(&mut self, error: ResponseTemplateError) -> ResponseTemplateError {
        self.poison = Some(error.clone());
        error
    }
}

/// Bytes retained after `consume_available` that count toward the pending
/// limit, and the unfinished field holding them, if there is one.
fn pending_state(template: &CompiledTemplate, text: &str) -> (usize, Option<OpenField>) {
    let mut cursor = 0usize;
    while cursor < text.len() {
        let Some((start, open_end, index)) = next_opener(template, text, cursor) else {
            // No opener has been recognized, so `consume_available` retains
            // this entire undecided suffix, including ignorable whitespace.
            // A delimiter's finite regex width does not bound those retained
            // bytes. Recognized field bodies retain their separate limits below.
            return (text.len() - cursor, None);
        };
        if !text[cursor..start].trim().is_empty() {
            return (text.len() - cursor, None);
        }
        let field = &template.fields[index];
        let tail = &text[open_end..];
        if let Some((body_end, close_len)) = earliest_close(tail, &field.closes) {
            cursor = open_end + body_end + close_len;
            continue;
        }
        // An unfinished field keeps its leading whitespace in `bytes` too.
        // Only complete fields took the draining `continue` above; their prefix
        // is no longer pending. Body bytes retain their independent body limit.
        let retained_prefix = start - cursor;
        let open = OpenField {
            field: index,
            retained_prefix,
            body_start: open_end,
            close_search_from: close_search_from(field, open_end, text.len()),
        };
        return match field.content {
            ContentKind::Text => (
                retained_prefix + longest_close_prefix_suffix(tail, &field.closes),
                Some(open),
            ),
            ContentKind::XmlInline => {
                let Some(tag) = &field.tag else {
                    return (retained_prefix + tail.len(), None);
                };
                let last_complete_tag = tag
                    .regex
                    .find_iter(tail)
                    .map(|found| found.end())
                    .max()
                    .unwrap_or(0);
                let unfinished = &tail[last_complete_tag..];
                let pending = match tag.bound {
                    DelimiterBound::Bounded(maximum) => unfinished.len().min(maximum),
                    DelimiterBound::Unbounded => unfinished.len(),
                };
                (retained_prefix + pending, Some(open))
            }
        };
    }
    (0, None)
}

/// The earliest field opener at or after `cursor`, as `(start, end, field)`.
/// Ties keep template field order.
fn next_opener(
    template: &CompiledTemplate,
    text: &str,
    cursor: usize,
) -> Option<(usize, usize, usize)> {
    let mut selected: Option<(usize, usize, usize)> = None;
    for (index, field) in template.fields.iter().enumerate() {
        if let Some(found) = field.open.find(&text[cursor..]) {
            let start = cursor + found.start();
            if selected.is_none_or(|(best_start, _, _)| start < best_start) {
                selected = Some((start, cursor + found.end(), index));
            }
        }
    }
    selected
}

/// Resume offset for the close search once `bytes[..valid_end]` holds no
/// complete close: a close may still start in its last `close length - 1`
/// bytes.
fn close_search_from(field: &CompiledField, body_start: usize, valid_end: usize) -> usize {
    let longest_close = field.closes.iter().map(String::len).max().unwrap_or(0);
    valid_end
        .saturating_sub(longest_close.saturating_sub(1))
        .max(body_start)
}

/// Largest UTF-8 scalar boundary at or before `index` in bytes whose prefix
/// up to `index` is valid UTF-8.
fn floor_char_boundary(bytes: &[u8], mut index: usize) -> usize {
    // Continuation bytes are 0b10xx_xxxx.
    while index > 0 && index < bytes.len() && bytes[index] & 0xC0 == 0x80 {
        index -= 1;
    }
    index
}

/// End of the valid UTF-8 prefix of `bytes`, given that `bytes[..checked]`
/// is already valid and ends on a scalar boundary. A trailing incomplete
/// scalar is held back; any other invalid byte is an error.
fn utf8_valid_end(
    template: &CompiledTemplate,
    bytes: &[u8],
    checked: usize,
) -> Result<usize, ResponseTemplateError> {
    match std::str::from_utf8(&bytes[checked..]) {
        Ok(_) => Ok(bytes.len()),
        Err(error) if error.error_len().is_none() => Ok(checked + error.valid_up_to()),
        Err(error) => Err(invalid_utf8(template, checked + error.valid_up_to())),
    }
}

fn utf8_str<'a>(
    template: &CompiledTemplate,
    bytes: &'a [u8],
    start: usize,
    end: usize,
) -> Result<&'a str, ResponseTemplateError> {
    std::str::from_utf8(&bytes[start..end])
        .map_err(|error| invalid_utf8(template, start + error.valid_up_to()))
}

fn invalid_utf8(template: &CompiledTemplate, at: usize) -> ResponseTemplateError {
    ResponseTemplateError::RuntimeFailure {
        model_name: template.model_name.clone(),
        field: "utf8".to_string(),
        limit: 0,
        reason: format!("invalid UTF-8 at byte {at}"),
    }
}

#[cfg(test)]
thread_local! {
    /// Retained bytes examined by `feed`, for the complexity regression test.
    static FEED_SCANNED_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn record_feed_scan(bytes: usize) {
    FEED_SCANNED_BYTES.with(|scanned| scanned.set(scanned.get() + bytes));
}

#[cfg(not(test))]
fn record_feed_scan(_bytes: usize) {}

/// Enforce opener, name, body and tag limits on the fields that
/// `consume_available` walks, including an unfinished trailing field. Text
/// inside a field body is body text and is not re-read as framing.
fn validate_runtime_limits(
    template: &CompiledTemplate,
    config: ParserConfig,
    text: &str,
) -> Result<(), ResponseTemplateError> {
    let mut cursor = 0usize;
    while cursor < text.len() {
        let Some((start, open_end, index)) = next_opener(template, text, cursor) else {
            break;
        };
        if !text[cursor..start].trim().is_empty() {
            break;
        }
        let field = &template.fields[index];
        ensure_pending_limit(template, config, open_end - start)?;
        if let Some(name) = field
            .open
            .captures(&text[start..])
            .filter(|captures| captures.get(0).is_some_and(|item| item.start() == 0))
            .and_then(|captures| captures.name("name"))
        {
            ensure_structured_limit(template, config, name.as_str().len())?;
        }
        let tail = &text[open_end..];
        let (body, next) = match earliest_close(tail, &field.closes) {
            Some((offset, close_len)) => (&tail[..offset], Some(open_end + offset + close_len)),
            None => {
                let held = longest_close_prefix_suffix(tail, &field.closes);
                (&tail[..tail.len() - held], None)
            }
        };
        ensure_body_limit(template, config, body.len())?;
        if let Some(tag) = &field.tag {
            validate_partial_tag_limits(template, config, tag, body)?;
            for tag_captures in tag.regex.captures_iter(body) {
                let whole_tag = tag_captures.get_match();
                ensure_pending_limit(template, config, whole_tag.as_str().len())?;
                if let Some(key) = tag_captures.name("key") {
                    ensure_structured_limit(template, config, key.as_str().len())?;
                }
                if let Some(value) = tag_captures.name("value") {
                    ensure_structured_limit(template, config, value.as_str().len())?;
                }
            }
        }
        match next {
            Some(next) => cursor = next,
            None => break,
        }
    }
    Ok(())
}

fn validate_partial_tag_limits(
    template: &CompiledTemplate,
    config: ParserConfig,
    tag: &crate::compiled::CompiledTag,
    body: &str,
) -> Result<(), ResponseTemplateError> {
    let consumed = tag
        .regex
        .find_iter(body)
        .map(|found| found.end())
        .max()
        .unwrap_or(0);
    let unfinished = &body[consumed..];
    if let Some(captures) = tag.partial_key.captures(unfinished) {
        if let Some(key) = captures.name("partial_key") {
            ensure_structured_limit(template, config, key.as_str().len())?;
        }
    }
    if let Some(captures) = tag.partial_value.captures(unfinished) {
        if let Some(value) = captures.name("partial_value") {
            let held_close = longest_literal_prefix_suffix(value.as_str(), &tag.value_close);
            ensure_structured_limit(
                template,
                config,
                value.as_str().len().saturating_sub(held_close),
            )?;
        }
    }
    Ok(())
}

fn longest_literal_prefix_suffix(text: &str, delimiter: &str) -> usize {
    let bytes = text.as_bytes();
    let delimiter = delimiter.as_bytes();
    (1..=bytes.len().min(delimiter.len()))
        .filter(|length| bytes[bytes.len() - length..] == delimiter[..*length])
        .max()
        .unwrap_or(0)
}

fn consume_available(
    template: &CompiledTemplate,
    config: ParserConfig,
    text: &str,
    finish_mode: Option<FinishMode>,
    seen: &mut BTreeSet<String>,
) -> Result<(ParseOutput, usize), ResponseTemplateError> {
    let eos = finish_mode.is_some();
    let mut output = ParseOutput::default();
    let mut cursor = 0usize;

    while cursor < text.len() {
        let selected = next_opener(template, text, cursor)
            .map(|(start, open_end, index)| (start, open_end, &template.fields[index]));
        let Some((start, open_end, field)) = selected else {
            if text[cursor..].trim().is_empty() {
                if eos {
                    cursor = text.len();
                }
                break;
            }
            if !eos {
                break;
            }
            return Err(runtime(
                template,
                "response",
                config.max_pending_bytes,
                "unrecognized trailing assistant output",
            ));
        };
        if !text[cursor..start].trim().is_empty() {
            if !eos {
                break;
            }
            return Err(runtime(
                template,
                &field.name,
                config.max_pending_bytes,
                "unrecognized bytes before field delimiter",
            ));
        }
        if !field.repeats && seen.contains(&field.name) {
            return Err(runtime(
                template,
                &field.name,
                config.max_body_bytes,
                "non-repeating field occurred more than once",
            ));
        }

        let captures = field
            .open
            .captures(&text[start..])
            .filter(|captures| captures.get(0).is_some_and(|item| item.start() == 0))
            .ok_or_else(|| {
                runtime(
                    template,
                    &field.name,
                    config.max_pending_bytes,
                    "field opener could not be recaptured",
                )
            })?;
        let tail = &text[open_end..];
        let (body_end, close_len) = match earliest_close(tail, &field.closes) {
            Some(close) => close,
            None if !eos => break,
            None if finish_mode == Some(FinishMode::LengthLimit)
                && field.content == ContentKind::Text =>
            {
                // At a length limit, a partial close delimiter is literal text.
                // Count it against the body limit below; do not manufacture a
                // structured/tool field.
                (tail.len(), 0)
            }
            None => {
                return Err(runtime(
                    template,
                    &field.name,
                    config.max_body_bytes,
                    "unterminated field",
                ));
            }
        };
        let body = &tail[..body_end];
        ensure_body_limit(template, config, body.len())?;

        match field.content {
            ContentKind::Text => match field.name.as_str() {
                "thinking" => output.thinking.push_str(body),
                "content" => output.content.push_str(body),
                _ => {
                    return Err(runtime(
                        template,
                        &field.name,
                        config.max_body_bytes,
                        "unsupported text output field",
                    ));
                }
            },
            ContentKind::XmlInline => {
                let name = captures
                    .name("name")
                    .ok_or_else(|| {
                        runtime(
                            template,
                            &field.name,
                            config.max_structured_field_bytes,
                            "missing tool name",
                        )
                    })?
                    .as_str();
                ensure_structured_limit(template, config, name.len())?;
                let arguments = parse_xml_inline(template, field, config, body)?;
                let transformed = field
                    .transform
                    .as_ref()
                    .map(|transform| apply_transform(&transform.0, name, &arguments))
                    .unwrap_or_else(|| Value::Object(arguments.clone()));
                output.tool_calls.push(ToolCall {
                    name: name.to_string(),
                    arguments,
                    transformed,
                });
            }
        }
        seen.insert(field.name.clone());
        cursor = open_end + body_end + close_len;
    }
    Ok((output, cursor))
}

fn parse_xml_inline(
    template: &CompiledTemplate,
    field: &CompiledField,
    config: ParserConfig,
    body: &str,
) -> Result<Map<String, Value>, ResponseTemplateError> {
    let tag = field.tag.as_ref().ok_or_else(|| {
        runtime(
            template,
            &field.name,
            config.max_body_bytes,
            "missing xml-inline tag parser",
        )
    })?;
    let mut arguments = Map::new();
    let mut cursor = 0usize;
    for captures in tag.regex.captures_iter(body) {
        let whole = captures.get_match();
        if !body[cursor..whole.start()].trim().is_empty() {
            return Err(runtime(
                template,
                &field.name,
                config.max_body_bytes,
                "reserved or malformed parameter delimiter",
            ));
        }
        let key = captures
            .name("key")
            .ok_or_else(|| {
                runtime(
                    template,
                    &field.name,
                    config.max_structured_field_bytes,
                    "missing parameter key",
                )
            })?
            .as_str();
        let raw_value = captures
            .name("value")
            .ok_or_else(|| {
                runtime(
                    template,
                    &field.name,
                    config.max_structured_field_bytes,
                    "missing parameter value",
                )
            })?
            .as_str();
        ensure_structured_limit(template, config, key.len())?;
        ensure_structured_limit(template, config, raw_value.len())?;
        let value = if tag.strip {
            raw_value.trim()
        } else {
            raw_value
        };
        arguments.insert(key.to_string(), Value::String(value.to_string()));
        cursor = whole.end();
    }
    if !body[cursor..].trim().is_empty() || (arguments.is_empty() && !body.trim().is_empty()) {
        return Err(runtime(
            template,
            &field.name,
            config.max_body_bytes,
            "reserved or malformed parameter delimiter",
        ));
    }
    Ok(arguments)
}

fn earliest_close(text: &str, closes: &[String]) -> Option<(usize, usize)> {
    closes
        .iter()
        .filter_map(|close| text.find(close).map(|offset| (offset, close.len())))
        .min_by_key(|(offset, _)| *offset)
}

fn longest_close_prefix_suffix(text: &str, closes: &[String]) -> usize {
    let bytes = text.as_bytes();
    closes
        .iter()
        .flat_map(|close| {
            let close = close.as_bytes();
            (1..=bytes.len().min(close.len()))
                .filter(move |&length| bytes[bytes.len() - length..] == close[..length])
        })
        .max()
        .unwrap_or(0)
}

fn ensure_structured_limit(
    template: &CompiledTemplate,
    config: ParserConfig,
    bytes: usize,
) -> Result<(), ResponseTemplateError> {
    if bytes > config.max_structured_field_bytes {
        return Err(ResponseTemplateError::StructuredFieldOverflow {
            model_name: template.model_name.clone(),
            field: "max_structured_field_bytes".to_string(),
            limit: config.max_structured_field_bytes,
        });
    }
    Ok(())
}

fn ensure_pending_limit(
    template: &CompiledTemplate,
    config: ParserConfig,
    bytes: usize,
) -> Result<(), ResponseTemplateError> {
    if bytes > config.max_pending_bytes {
        return Err(ResponseTemplateError::PendingOverflow {
            model_name: template.model_name.clone(),
            field: "max_pending_bytes".to_string(),
            limit: config.max_pending_bytes,
        });
    }
    Ok(())
}

fn ensure_body_limit(
    template: &CompiledTemplate,
    config: ParserConfig,
    bytes: usize,
) -> Result<(), ResponseTemplateError> {
    if bytes > config.max_body_bytes {
        return Err(ResponseTemplateError::StructuredFieldOverflow {
            model_name: template.model_name.clone(),
            field: "max_body_bytes".to_string(),
            limit: config.max_body_bytes,
        });
    }
    Ok(())
}

fn runtime(
    template: &CompiledTemplate,
    field: &str,
    limit: usize,
    reason: impl Into<String>,
) -> ResponseTemplateError {
    ResponseTemplateError::RuntimeFailure {
        model_name: template.model_name.clone(),
        field: field.to_string(),
        limit,
        reason: reason.into(),
    }
}

fn apply_defaults(
    template: &CompiledTemplate,
    seen: &BTreeSet<String>,
    output: &mut ParseOutput,
) -> Result<(), ResponseTemplateError> {
    for (field, target) in [
        ("thinking", &mut output.thinking),
        ("content", &mut output.content),
    ] {
        if seen.contains(field) {
            continue;
        }
        if let Some(value) = template.defaults.get(field) {
            let value = value
                .as_str()
                .ok_or_else(|| ResponseTemplateError::InvalidTemplate {
                    model_name: template.model_name.clone(),
                    field: field.to_string(),
                    limit: 0,
                    reason: "text-field default must be a string".to_string(),
                })?;
            target.push_str(value);
        }
    }
    Ok(())
}

fn apply_transform(template: &Value, name: &str, content: &Map<String, Value>) -> Value {
    match template {
        Value::String(value) if value == "{name}" => Value::String(name.to_string()),
        Value::String(value) if value == "{content}" => Value::Object(content.clone()),
        Value::String(value) => Value::String(value.replace("{name}", name)),
        Value::Array(values) => Value::Array(
            values
                .iter()
                .map(|value| apply_transform(value, name, content))
                .collect(),
        ),
        Value::Object(values) => Value::Object(
            values
                .iter()
                .map(|(key, value)| (key.clone(), apply_transform(value, name, content)))
                .collect(),
        ),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const PREFIX: &str = "<|start|>assistant";
    const OPEN: &str = "<|channel|>final<|message|>";
    const CLOSE: &str = "<|return|>";
    const CALL_OPEN: &str = "to=f<|message|>";
    const CALL_CLOSE: &str = "<|call|>";

    fn parser(config: ParserConfig) -> ResponseTemplateParser {
        let template = json!({
            "start_anchor_pattern": r"<\|start\|>assistant",
            "fields": {
                "thinking": {
                    "open_pattern": r"<\|channel\|>analysis<\|message\|>",
                    "close": "<|end|>",
                    "content": "text"
                },
                "content": {
                    "open_pattern": r"<\|channel\|>final<\|message\|>",
                    "close": [CLOSE, "<|end|>"],
                    "content": "text"
                },
                "tool_calls": {
                    "open_pattern": r"to=(?P<name>[^<]+)<\|message\|>",
                    "close": "<|call|>",
                    "content": "xml-inline",
                    "content_args": {
                        "tag_pattern": r#"<arg name="(?P<key>[^"]+)">(?P<value>.*?)</arg>"#,
                        "value_parser": {"name": "text"}
                    },
                    "repeats": true,
                    "transform": {"name": "{name}", "arguments": "{content}"}
                }
            }
        });
        ResponseTemplateParser::from_json("scan-test", &template, config).unwrap()
    }

    /// Feed `wire` in `chunk`-byte pieces and return the output together with
    /// the retained bytes `feed` examined.
    fn feed_counting(
        parser: &ResponseTemplateParser,
        wire: &str,
        chunk: usize,
    ) -> (ParseOutput, usize) {
        FEED_SCANNED_BYTES.with(|scanned| scanned.set(0));
        let mut stream = parser.stream(PREFIX);
        let mut output = ParseOutput::default();
        for piece in wire.as_bytes().chunks(chunk) {
            output.merge(stream.feed(piece).unwrap());
        }
        output.merge(stream.finish().unwrap());
        (output, FEED_SCANNED_BYTES.with(std::cell::Cell::get))
    }

    #[test]
    fn open_text_field_feeds_scan_only_new_bytes() {
        let parser = parser(ParserConfig::default());
        // Byte-at-a-time and chunked delivery of long bodies, with multi-byte
        // scalars split across feeds.
        for (body_len, chunk) in [(16 * 1024, 1), (1024 * 1024, 4096)] {
            let body = "é".repeat(body_len / 2);
            let wire = format!("{OPEN}{body}{CLOSE}");
            let (output, scanned) = feed_counting(&parser, &wire, chunk);
            assert_eq!(output.content, body);
            // Each feed examines its own bytes plus a close-length overlap;
            // the opener and the closing feed scan the retained field once.
            let feeds = wire.len().div_ceil(chunk);
            let bound = feeds * (chunk + CLOSE.len()) + 2 * wire.len();
            assert!(
                scanned <= bound,
                "scanned {scanned} bytes for a {}-byte field in {feeds} feeds (bound {bound})",
                wire.len()
            );
        }
    }

    #[test]
    fn open_text_field_close_split_across_feeds_matches_one_chunk() {
        let parser = parser(ParserConfig::default());
        let wire = format!("{OPEN}a<|ret{CLOSE}<|channel|>analysis<|message|>t<|end|>");
        let expected = parser.parse_complete(PREFIX, &wire).unwrap();
        assert_eq!(expected.content, "a<|ret");
        assert_eq!(expected.thinking, "t");
        for chunk in 1..=wire.len() {
            let (output, _) = feed_counting(&parser, &wire, chunk);
            assert_eq!(output, expected, "chunk size {chunk}");
        }
    }

    #[test]
    fn open_text_field_limit_failure_restores_committed_bytes() {
        let parser = parser(ParserConfig {
            max_pending_bytes: 64,
            max_structured_field_bytes: 64,
            max_body_bytes: 8,
        });
        let mut stream = parser.stream(PREFIX);
        assert!(stream.feed(OPEN.as_bytes()).unwrap().is_empty());
        assert!(stream.open_field.is_some());
        assert!(stream.feed(b"12345678<|ret").unwrap().is_empty());
        let committed = stream.pending_bytes();
        let error = stream.feed(b"x").unwrap_err();
        assert!(matches!(
            &error,
            ResponseTemplateError::StructuredFieldOverflow { field, limit: 8, .. }
                if field == "max_body_bytes"
        ));
        assert_eq!(stream.pending_bytes(), committed);
        assert_eq!(stream.feed(CLOSE.as_bytes()).unwrap_err(), error);
        assert_eq!(stream.finish().unwrap_err(), error);
    }

    #[test]
    fn open_tool_call_field_feeds_scan_only_new_bytes() {
        let parser = parser(ParserConfig::default());
        // One long argument value, then complete arguments, delivered byte by
        // byte and in chunks, with multi-byte scalars split across feeds.
        for (value_len, chunk) in [(16 * 1024, 1), (1024 * 1024, 4096)] {
            let value = "é".repeat(value_len / 2);
            let wire = format!(
                "{CALL_OPEN}<arg name=\"long\">{value}</arg><arg name=\"k\">v</arg>{CALL_CLOSE}"
            );
            let (output, scanned) = feed_counting(&parser, &wire, chunk);
            assert_eq!(output.tool_calls.len(), 1);
            assert_eq!(output.tool_calls[0].arguments["long"], value.as_str());
            assert_eq!(output.tool_calls[0].arguments["k"], "v");
            let feeds = wire.len().div_ceil(chunk);
            let bound = feeds * (chunk + CALL_CLOSE.len()) + 2 * wire.len();
            assert!(
                scanned <= bound,
                "scanned {scanned} bytes for a {}-byte field in {feeds} feeds (bound {bound})",
                wire.len()
            );
        }
    }

    #[test]
    fn open_tool_call_field_near_a_limit_is_checked_in_full() {
        let tags = (0..16)
            .map(|index| format!("<arg name=\"k{index}\">{index}</arg>"))
            .collect::<String>();
        let pending_only = parser(ParserConfig {
            max_pending_bytes: 48,
            max_structured_field_bytes: 4096,
            max_body_bytes: 4096,
        });
        // Complete tags are not pending, so a body far beyond the pending
        // limit is accepted at every split.
        let wire = format!("{CALL_OPEN}{tags}{CALL_CLOSE}");
        let expected = pending_only.parse_complete(PREFIX, &wire).unwrap();
        assert_eq!(expected.tool_calls[0].arguments.len(), 16);
        for chunk in 1..=wire.len() {
            let (output, _) = feed_counting(&pending_only, &wire, chunk);
            assert_eq!(output, expected, "chunk size {chunk}");
        }

        // An unfinished tag is pending while it streams, and an unfinished
        // value counts against the structured limit.
        let structured_too = parser(ParserConfig {
            max_pending_bytes: 48,
            max_structured_field_bytes: 24,
            max_body_bytes: 4096,
        });
        let unfinished = format!("{CALL_OPEN}{tags}<arg name=\"v\">{}", "x".repeat(40));
        for (parser, variant, limit_field, limit) in [
            (&pending_only, "PendingOverflow", "max_pending_bytes", 48),
            (
                &structured_too,
                "StructuredFieldOverflow",
                "max_structured_field_bytes",
                24,
            ),
        ] {
            for chunk in 1..=unfinished.len() {
                let mut stream = parser.stream(PREFIX);
                let error = unfinished
                    .as_bytes()
                    .chunks(chunk)
                    .find_map(|piece| stream.feed(piece).err())
                    .unwrap_or_else(|| panic!("chunk size {chunk}: no overflow"));
                assert_eq!(
                    (error.variant_name(), error.field(), error.limit()),
                    (variant, limit_field, limit),
                    "chunk size {chunk}"
                );
            }
        }
    }
}
