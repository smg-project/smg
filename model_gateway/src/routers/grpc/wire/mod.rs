//! The wire: what a vendor changes in the bytes a successful answer leaves
//! in, decided once per request.
//!
//! The provider profile (`openai_protocol::profile`) is the request-side
//! contract: what a vendor's requests may say and how they are normalized.
//! This module is the response side: the shape of a chat stream and the
//! re-slicing of a relayed stream. A [`Wire`] is selected from the profile
//! alone ([`Wire::of`]): in `ChatResponseSpec::for_request` for the stream,
//! and at the HTTP relay for its re-chunking. It is a fieldless `Copy` enum,
//! so carrying it costs nothing, and every vendor rule is one `match` whose
//! OpenAI arm is the code every other model runs. The chat stream resolves
//! its wire once, when it opens, into a [`StreamShape`]; per chunk it reads
//! plain fields, as before.
//!
//! One file per vendor holds that vendor's rules: [`deepseek`] and
//! [`minimax`]. The OpenAI wire has no file: it is the unconditional code at
//! each hook. Outside the object: the request side (the profile), the error
//! bodies and the SSE error events inside a 200 stream.

use openai_protocol::{common::StreamOptions, profile::ProviderProfile};

use crate::routers::common::sse_rechunk::RechunkPolicy;

pub(crate) mod deepseek;
pub(crate) mod minimax;

/// The vendor whose shapes a request's successful answer takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Wire {
    /// The OpenAI API's shapes: every model without a vendor wire of its own.
    OpenAi,
    /// DeepSeek's ([`deepseek`]).
    DeepSeek,
    /// MiniMax's ([`minimax`]).
    Minimax,
}

impl Wire {
    /// The wire of the models `profile` serves, by the profile alone.
    pub(crate) fn of(profile: ProviderProfile) -> Self {
        match profile {
            ProviderProfile::DeepSeek => Self::DeepSeek,
            ProviderProfile::Minimax => Self::Minimax,
            ProviderProfile::Generic
            | ProviderProfile::OpenAi
            | ProviderProfile::Kimi
            | ProviderProfile::Zai => Self::OpenAi,
        }
    }

    /// [`Self::of`] the profile of `model`, the model the client asked for
    /// (as request validation selects it, not the alias-resolved id).
    pub(crate) fn for_model(model: &str) -> Self {
        Self::of(ProviderProfile::for_model(model))
    }

    /// The chat stream's rules for a request with `options`, resolved once
    /// when the stream opens.
    pub(crate) fn stream(self, options: Option<&StreamOptions>) -> StreamShape {
        match self {
            Self::OpenAi | Self::Minimax => StreamShape::openai(options),
            Self::DeepSeek => deepseek::stream(options),
        }
    }

    /// How a chat stream's SSE events are re-sliced on their way out
    /// (`routers::common::sse_rechunk`), on the gRPC lane and the regular
    /// HTTP relay alike: the wire's numbers, or `None` for a wire whose
    /// events go out as the engine made them.
    pub(crate) fn rechunk_policy(self) -> Option<RechunkPolicy> {
        match self {
            Self::Minimax => Some(minimax::RECHUNK),
            Self::OpenAi | Self::DeepSeek => None,
        }
    }
}

/// What a chat stream decides once, when it opens, from its wire and the
/// request's `stream_options`; the stream keeps these as plain fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StreamShape {
    /// Whether the stream reports its final usage at all: the client's
    /// `include_usage` on the OpenAI wire, always on DeepSeek's.
    pub reports_usage: bool,
    /// Whether a chunk that carries no usage writes `"usage": null`: when
    /// the client asked for usage, as the public API does, on every wire.
    pub usage_null_placeholder: bool,
    /// Whether every chunk carries the running usage
    /// (`continuous_usage_stats` with `include_usage`); never on DeepSeek's
    /// wire.
    pub continuous_usage: bool,
    /// Which chunk carries the final usage.
    pub final_usage_on: FinalUsageOn,
}

impl StreamShape {
    /// The OpenAI API's stream: the usage goes where, and only when, the
    /// client asked for it.
    pub(crate) fn openai(options: Option<&StreamOptions>) -> Self {
        let reports_usage = client_asked_for_usage(options);
        Self {
            reports_usage,
            usage_null_placeholder: reports_usage,
            continuous_usage: reports_usage
                && options.is_some_and(|opts| opts.continuous_usage_stats.unwrap_or(false)),
            final_usage_on: FinalUsageOn::UsageChunk,
        }
    }
}

/// `stream_options.include_usage` as the client sent it.
pub(crate) fn client_asked_for_usage(options: Option<&StreamOptions>) -> bool {
    options.is_some_and(|opts| opts.include_usage.unwrap_or(false))
}

/// Which chunk carries a stream's final usage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FinalUsageOn {
    /// The OpenAI API's opt-in usage-only chunk after the finish chunks.
    UsageChunk,
    /// The last finish chunk, and no usage-only chunk after it (DeepSeek's).
    LastFinish,
}

impl FinalUsageOn {
    /// Whether the finish chunk at `position`, counted from one, of `count`
    /// carries the final usage.
    pub(crate) fn carries(self, position: usize, count: usize) -> bool {
        match self {
            Self::UsageChunk => false,
            Self::LastFinish => position == count,
        }
    }

    /// Whether the stream closes with the usage-only chunk.
    pub(crate) fn sends_usage_chunk(self) -> bool {
        match self {
            Self::UsageChunk => true,
            Self::LastFinish => false,
        }
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/routers/grpc/wire/tests.rs"]
mod tests;
