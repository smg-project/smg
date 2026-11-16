//! DeepSeek's wire (deepseek-provider-verifier): a chat stream always
//! reports its final usage on the last finish chunk and sends no usage-only
//! chunk after it, and the running usage of `continuous_usage_stats` is
//! never sent. Everything else is the OpenAI wire: when the client asked for
//! usage, the chunks before the last write `"usage": null`.

use openai_protocol::common::StreamOptions;

use super::{FinalUsageOn, StreamShape};

/// The stream rules of a DeepSeek request with `options`.
pub(super) fn stream(options: Option<&StreamOptions>) -> StreamShape {
    StreamShape {
        reports_usage: true,
        continuous_usage: false,
        final_usage_on: FinalUsageOn::LastFinish,
        ..StreamShape::openai(options)
    }
}
