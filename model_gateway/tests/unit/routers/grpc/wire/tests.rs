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
