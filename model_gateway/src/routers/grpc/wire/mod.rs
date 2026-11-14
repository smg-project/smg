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
mod tests {
    use super::*;

    fn options(include_usage: Option<bool>, continuous: Option<bool>) -> StreamOptions {
        StreamOptions {
            include_usage,
            continuous_usage_stats: continuous,
            ..StreamOptions::default()
        }
    }

    /// The profile alone selects the wire: DeepSeek and MiniMax their own,
    /// every other profile the OpenAI wire.
    #[test]
    fn wire_follows_the_profile_alone() {
        assert_eq!(Wire::of(ProviderProfile::DeepSeek), Wire::DeepSeek);
        assert_eq!(Wire::of(ProviderProfile::Minimax), Wire::Minimax);
        for profile in [
            ProviderProfile::Generic,
            ProviderProfile::OpenAi,
            ProviderProfile::Kimi,
            ProviderProfile::Zai,
        ] {
            assert_eq!(Wire::of(profile), Wire::OpenAi, "{profile:?}");
        }
        for (model, wire) in [
            ("deepseek-flash", Wire::DeepSeek),
            ("MiniMax-M3", Wire::Minimax),
            ("gpt-4o", Wire::OpenAi),
            ("self-hosted-model", Wire::OpenAi),
        ] {
            assert_eq!(Wire::for_model(model), wire, "{model}");
        }
    }

    /// The OpenAI stream reads the client's options: no usage and no
    /// placeholder unless asked, the running usage only with `include_usage`.
    #[test]
    fn openai_stream_reads_the_clients_options() {
        let shape = |reports_usage, continuous_usage| StreamShape {
            reports_usage,
            usage_null_placeholder: reports_usage,
            continuous_usage,
            final_usage_on: FinalUsageOn::UsageChunk,
        };
        for wire in [Wire::OpenAi, Wire::Minimax] {
            assert_eq!(wire.stream(None), shape(false, false), "{wire:?}");
            for (include, continuous, expected) in [
                (Some(true), Some(true), shape(true, true)),
                (Some(true), None, shape(true, false)),
                (Some(false), Some(true), shape(false, false)),
                (None, Some(true), shape(false, false)),
            ] {
                assert_eq!(
                    wire.stream(Some(&options(include, continuous))),
                    expected,
                    "{wire:?} {include:?} {continuous:?}"
                );
            }
        }
        assert!(FinalUsageOn::UsageChunk.sends_usage_chunk());
        assert!(!FinalUsageOn::UsageChunk.carries(2, 2));
    }

    /// DeepSeek closes the usage on the last finish chunk whatever the
    /// client asked, writes the null placeholder only when it asked for
    /// usage, and never sends the running usage or a usage-only chunk.
    #[test]
    fn deepseek_stream_closes_on_the_last_finish_chunk() {
        for (include, placeholder) in [(None, false), (Some(false), false), (Some(true), true)] {
            let shape = Wire::DeepSeek.stream(Some(&options(include, Some(true))));
            assert_eq!(
                shape,
                StreamShape {
                    reports_usage: true,
                    usage_null_placeholder: placeholder,
                    continuous_usage: false,
                    final_usage_on: FinalUsageOn::LastFinish,
                },
                "{include:?}"
            );
        }
        assert!(Wire::DeepSeek.stream(None).reports_usage);
        assert!(!FinalUsageOn::LastFinish.sends_usage_chunk());
        assert!(FinalUsageOn::LastFinish.carries(2, 2));
        assert!(!FinalUsageOn::LastFinish.carries(1, 2));
    }

    /// Only MiniMax re-slices its stream, by its own numbers.
    #[test]
    fn rechunk_names_its_vendor() {
        for wire in [Wire::OpenAi, Wire::DeepSeek, Wire::Minimax] {
            assert_eq!(
                wire.rechunk_policy().is_some(),
                wire == Wire::Minimax,
                "{wire:?}"
            );
        }
        assert_eq!(Wire::Minimax.rechunk_policy(), Some(minimax::RECHUNK));
    }
}
