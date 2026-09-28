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

impl ResponseTemplateParser {
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

    pub fn delimiter_metadata(&self) -> &[crate::DelimiterMetadata] {
        &self.compiled.metadata
    }

    pub fn stream(&self, rendered_prompt_prefix: impl Into<String>) -> StreamingParser {
        StreamingParser {
            compiled: Arc::clone(&self.compiled),
            config: self.config,
            rendered_prompt_prefix: rendered_prompt_prefix.into(),
            anchor_validated: false,
            bytes: Vec::new(),
            seen_fields: BTreeSet::new(),
            poison: None,
            finished: false,
        }
    }

    pub fn parse_complete(
        &self,
        rendered_prompt_prefix: &str,
        decoded_output: &str,
    ) -> Result<ParseOutput, ResponseTemplateError> {
        let mut stream = self.stream(rendered_prompt_prefix);
        let mut output = stream.feed(decoded_output.as_bytes())?;
        output.merge(stream.finish()?);
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
    seen_fields: BTreeSet<String>,
    poison: Option<ResponseTemplateError>,
    finished: bool,
}

impl StreamingParser {
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

        let mut candidate = self.bytes.clone();
        candidate.extend_from_slice(chunk);
        let (valid, incomplete_utf8_bytes) = match std::str::from_utf8(&candidate) {
            Ok(text) => (text, 0),
            Err(error) if error.error_len().is_none() => {
                #[expect(
                    clippy::expect_used,
                    reason = "Utf8Error guarantees that the prefix before valid_up_to is valid UTF-8"
                )]
                let valid = std::str::from_utf8(&candidate[..error.valid_up_to()])
                    .expect("Utf8Error valid_up_to prefix is valid");
                (valid, candidate.len().saturating_sub(error.valid_up_to()))
            }
            Err(error) => {
                return Err(self.poison(ResponseTemplateError::RuntimeFailure {
                    model_name: self.compiled.model_name.clone(),
                    field: "utf8".to_string(),
                    limit: 0,
                    reason: format!("invalid UTF-8 at byte {}", error.valid_up_to()),
                }));
            }
        };
        if let Err(error) = validate_runtime_limits(&self.compiled, self.config, valid) {
            return Err(self.poison(error));
        }
        let mut staged_seen = self.seen_fields.clone();
        let (output, consumed) =
            match consume_available(&self.compiled, self.config, valid, false, &mut staged_seen) {
                Ok(result) => result,
                Err(error) => return Err(self.poison(error)),
            };
        let pending_bytes =
            pending_byte_len(&self.compiled, &valid[consumed..]) + incomplete_utf8_bytes;
        if pending_bytes > self.config.max_pending_bytes {
            return Err(self.poison(ResponseTemplateError::PendingOverflow {
                model_name: self.compiled.model_name.clone(),
                field: "max_pending_bytes".to_string(),
                limit: self.config.max_pending_bytes,
            }));
        }
        self.bytes = candidate[consumed..].to_vec();
        self.seen_fields = staged_seen;
        Ok(output)
    }

    pub fn finish(&mut self) -> Result<ParseOutput, ResponseTemplateError> {
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
        let (mut output, consumed) =
            match consume_available(&self.compiled, self.config, text, true, &mut staged_seen) {
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
        self.seen_fields = staged_seen;
        apply_defaults(&self.compiled, &self.seen_fields, &mut output)?;
        self.finished = true;
        Ok(output)
    }

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

fn pending_byte_len(template: &CompiledTemplate, text: &str) -> usize {
    let mut cursor = 0usize;
    while cursor < text.len() {
        let selected = template
            .fields
            .iter()
            .filter_map(|field| {
                field
                    .open
                    .find(&text[cursor..])
                    .map(|found| (cursor + found.start(), cursor + found.end(), field))
            })
            .min_by_key(|(start, _, _)| *start);
        let Some((start, open_end, field)) = selected else {
            // No opener has been recognized, so `consume_available` retains
            // this entire undecided suffix, including ignorable whitespace.
            // A delimiter's finite regex width does not bound those retained
            // bytes. Recognized field bodies retain their separate limits below.
            return text.len() - cursor;
        };
        if !text[cursor..start].trim().is_empty() {
            return text.len() - cursor;
        }
        let tail = &text[open_end..];
        if let Some((body_end, close_len)) = earliest_close(tail, &field.closes) {
            cursor = open_end + body_end + close_len;
            continue;
        }
        // An unfinished field keeps its leading whitespace in `bytes` too.
        // Only complete fields took the draining `continue` above; their prefix
        // is no longer pending. Body bytes retain their independent body limit.
        let retained_prefix = start - cursor;
        let pending_field = match field.content {
            ContentKind::Text => longest_close_prefix_suffix(tail, &field.closes),
            ContentKind::XmlInline => {
                let Some(tag) = &field.tag else {
                    return retained_prefix + tail.len();
                };
                let last_complete_tag = tag
                    .regex
                    .find_iter(tail)
                    .map(|found| found.end())
                    .max()
                    .unwrap_or(0);
                let unfinished = &tail[last_complete_tag..];
                match tag.bound {
                    DelimiterBound::Bounded(maximum) => unfinished.len().min(maximum),
                    DelimiterBound::Unbounded => unfinished.len(),
                }
            }
        };
        return retained_prefix + pending_field;
    }
    0
}

fn validate_runtime_limits(
    template: &CompiledTemplate,
    config: ParserConfig,
    text: &str,
) -> Result<(), ResponseTemplateError> {
    for field in &template.fields {
        for captures in field.open.captures_iter(text) {
            let whole = captures.get_match();
            ensure_pending_limit(template, config, whole.as_str().len())?;
            if let Some(name) = captures.name("name") {
                ensure_structured_limit(template, config, name.as_str().len())?;
            }
            let tail = &text[whole.end()..];
            let body = match earliest_close(tail, &field.closes) {
                Some((offset, _)) => &tail[..offset],
                None => {
                    let held = longest_close_prefix_suffix(tail, &field.closes);
                    &tail[..tail.len() - held]
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
    eos: bool,
    seen: &mut BTreeSet<String>,
) -> Result<(ParseOutput, usize), ResponseTemplateError> {
    let mut output = ParseOutput::default();
    let mut cursor = 0usize;

    while cursor < text.len() {
        let mut selected: Option<(usize, usize, &CompiledField)> = None;
        for field in &template.fields {
            if let Some(found) = field.open.find(&text[cursor..]) {
                let start = cursor + found.start();
                let end = cursor + found.end();
                if selected
                    .as_ref()
                    .is_none_or(|(best_start, _, _)| start < *best_start)
                {
                    selected = Some((start, end, field));
                }
            }
        }
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
        let Some((body_end, close_len)) = earliest_close(tail, &field.closes) else {
            if !eos {
                break;
            }
            return Err(runtime(
                template,
                &field.name,
                config.max_body_bytes,
                "unterminated field",
            ));
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
