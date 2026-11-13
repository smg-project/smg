//! Single-owner decode input gate. The actor drives native sends in returned order.

use std::time::Instant;

use prost::Message;
use smg_grpc_client::{
    sglang_managed::{
        budget::Deadline, proto, validate_binding, validate_manifest, Limits, VERSION,
    },
    sglang_proto,
};
use tonic::Status;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    WaitingInput,
    InputBound,
    NativeSubmitting,
    NativeSubmitted,
    Terminal,
}

#[derive(Debug)]
pub enum NativeInput {
    Text(sglang_proto::GenerateRequest),
    Prepared {
        generation: sglang_proto::GenerateRequest,
        manifest: Box<proto::PreparedDecodeInput>,
    },
}

/// A value cannot be cloned: exactly one actor owns submission and ordered abort.
pub struct DecodeGate {
    start: proto::DecodeStart,
    binding: proto::InputBinding,
    deadline: Deadline,
    limits: Limits,
    state: State,
    bound_digest: Option<Vec<u8>>,
    prepared: Option<proto::PreparedDecodeInput>,
    abort_after_submit: bool,
}

impl DecodeGate {
    /// Admission must reserve bounded WaitingInput memory before publishing Accepted.
    pub fn register(
        start: proto::DecodeStart,
        instance: &proto::Instance,
        deadline: Deadline,
        limits: Limits,
        now: Instant,
    ) -> Result<Self, Status> {
        deadline.remaining(now)?;
        let binding = start
            .binding
            .clone()
            .ok_or_else(|| Status::invalid_argument("missing input binding"))?;
        validate_binding(&binding)?;
        let operation = binding
            .decode
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("missing operation"))?;
        if start.version != VERSION || operation.target.as_ref() != Some(instance) {
            return Err(Status::failed_precondition(
                "managed decode instance/version mismatch",
            ));
        }
        if start.encoded_len() > limits.manifest_bytes {
            return Err(Status::resource_exhausted(
                "managed Start exceeds byte limit",
            ));
        }
        let generation = start
            .generation
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("missing generation options"))?;
        if generation.mm_inputs.is_some()
            || generation.media_identity.is_some()
            || !generation.extra_mm_inputs.is_empty()
            || !generation.input_embeds.is_empty()
            || generation.sampling_params.as_ref().is_some_and(|p| p.n > 1)
        {
            return Err(Status::invalid_argument(
                "managed D refuses media, embeddings and n > 1",
            ));
        }
        let tokens = generation
            .tokenized
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("missing tokenized request"))?;
        if tokens.input_ids.len() > limits.tokens
            || (start.deferred_from_prefill && !tokens.input_ids.is_empty())
            || (!start.deferred_from_prefill && tokens.input_ids.is_empty())
        {
            return Err(Status::invalid_argument(
                "invalid managed decode input mode",
            ));
        }
        let state = if start.deferred_from_prefill {
            State::WaitingInput
        } else {
            State::InputBound
        };
        Ok(Self {
            start,
            binding,
            deadline,
            limits,
            state,
            bound_digest: None,
            prepared: None,
            abort_after_submit: false,
        })
    }

    /// The binding is checked even on duplicate frames; a conflicting duplicate fails.
    pub fn bind(
        &mut self,
        input: proto::PreparedDecodeInput,
        now: Instant,
    ) -> Result<bool, Status> {
        self.deadline.remaining(now)?;
        if self.state == State::Terminal || self.abort_after_submit {
            return Err(Status::cancelled("managed decode is terminal"));
        }
        if !self.start.deferred_from_prefill {
            return Err(Status::invalid_argument(
                "inline text does not accept a manifest",
            ));
        }
        validate_manifest(&input, &self.binding, self.limits)?;
        if let Some(digest) = &self.bound_digest {
            return if digest == &input.integrity_digest {
                Ok(false)
            } else {
                Err(Status::already_exists("conflicting managed input binding"))
            };
        }
        if self.state != State::WaitingInput {
            return Err(Status::failed_precondition("managed input gate is closed"));
        }
        self.bound_digest = Some(input.integrity_digest.clone());
        self.prepared = Some(input);
        self.state = State::InputBound;
        Ok(true)
    }

    /// Consume the input once before issuing Generate on the rank's ordered channel.
    pub fn take_submission(&mut self, now: Instant) -> Result<Option<NativeInput>, Status> {
        self.deadline.remaining(now)?;
        if self.state != State::InputBound {
            return Ok(None);
        }
        let generation = self
            .start
            .generation
            .take()
            .ok_or_else(|| Status::internal("managed generation consumed"))?;
        let input = if self.start.deferred_from_prefill {
            NativeInput::Prepared {
                generation,
                manifest: Box::new(
                    self.prepared
                        .take()
                        .ok_or_else(|| Status::internal("managed manifest missing"))?,
                ),
            }
        } else {
            NativeInput::Text(generation)
        };
        self.state = State::NativeSubmitting;
        Ok(Some(input))
    }

    /// Returns whether Abort must follow Generate on that same native channel.
    pub fn submitted(&mut self) -> Result<bool, Status> {
        if self.state != State::NativeSubmitting {
            return Err(Status::failed_precondition(
                "unexpected native submit completion",
            ));
        }
        self.state = if self.abort_after_submit {
            State::Terminal
        } else {
            State::NativeSubmitted
        };
        Ok(self.abort_after_submit)
    }

    /// Cancel during send defers Abort until send completes; it never reopens the gate.
    pub fn cancel(&mut self) -> bool {
        self.prepared = None;
        match self.state {
            State::NativeSubmitting => {
                self.abort_after_submit = true;
                false
            }
            State::NativeSubmitted => {
                self.state = State::Terminal;
                true
            }
            _ => {
                self.state = State::Terminal;
                false
            }
        }
    }

    /// Normal request half-close is valid only after complete input was bound.
    pub fn half_close(&mut self) -> Result<(), Status> {
        if self.state == State::WaitingInput {
            self.cancel();
            return Err(Status::invalid_argument(
                "decode stream closed before input binding",
            ));
        }
        Ok(())
    }

    /// State is observational; it is not a claim about KV allocation or execution.
    pub fn state(&self) -> State {
        self.state
    }
}
