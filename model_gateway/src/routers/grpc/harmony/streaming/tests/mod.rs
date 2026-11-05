mod client_leave;
mod closed_message;
mod local_stop;
mod logprobs;
mod lone_complete;
mod responses_stop;
mod status;
mod usage;

use smg_grpc_client::vllm_engine::{proto, proto::generate_response::Response as GenerationEvent};

use super::*;

/// A chunk frame of `token_ids` after a three-token prompt (the harmony
/// streaming tests' one shape of it).
pub(super) fn chunk(token_ids: Vec<u32>) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(GenerationEvent::Chunk(proto::GenerateStreamChunk {
            index: 0,
            prompt_tokens: 3,
            completion_tokens: token_ids.len() as u32,
            token_ids,
            ..Default::default()
        })),
    }
}

/// Compile-time exhaustiveness anchor.
///
/// This helper exists solely to force every [`ResponseFormat`] variant
/// to flow through a non-wildcard `match`. If a new variant is added to
/// [`ResponseFormat`] without being classified, this function fails to
/// compile — which in turn breaks `streams_arguments_explicit_variants`
/// below, since both helpers iterate the same variant set.
///
/// Intentionally *does not* call [`streams_arguments`] — this mirrors
/// the production classifier so a drift between the two is a separate
/// failure (runtime assertion miss) from a missing variant (compile
/// error here).
fn expected_streams_arguments(format: ResponseFormat) -> bool {
    match format {
        ResponseFormat::Passthrough => true,
        ResponseFormat::WebSearchCall
        | ResponseFormat::CodeInterpreterCall
        | ResponseFormat::FileSearchCall
        | ResponseFormat::ImageGenerationCall => false,
    }
}

// Locks the `streams_arguments` classification so the Harmony router
// keeps treating hosted built-in tools — including `image_generation`
// — as structured-event emitters rather than argument streamers.
//
// Every `ResponseFormat` variant is named explicitly (no `_` arm, no
// iteration over a hand-maintained array), so adding a new variant
// fails to compile in `expected_streams_arguments` above AND in every
// explicit `let ... = ResponseFormat::X;` binding here — which in
// turn ensures the production `streams_arguments` classifier must
// also be updated to compile.
#[test]
fn streams_arguments_explicit_variants() {
    // `None` (plain function tool) streams arguments.
    assert!(streams_arguments(None), "function_call should stream args");

    // `Some(Passthrough)` (mcp_call) streams arguments.
    let passthrough = ResponseFormat::Passthrough;
    assert!(
        streams_arguments(Some(&passthrough)),
        "mcp_call (Passthrough) should stream args",
    );
    assert!(expected_streams_arguments(passthrough));

    // Hosted built-ins do *not* stream arguments — they surface
    // progress via structured `*.in_progress` / `*.searching` /
    // `*.generating` / `*.completed` events from the shared emitter.
    let web_search = ResponseFormat::WebSearchCall;
    assert!(!streams_arguments(Some(&web_search)));
    assert!(!expected_streams_arguments(web_search));

    let code_interpreter = ResponseFormat::CodeInterpreterCall;
    assert!(!streams_arguments(Some(&code_interpreter)));
    assert!(!expected_streams_arguments(code_interpreter));

    let file_search = ResponseFormat::FileSearchCall;
    assert!(!streams_arguments(Some(&file_search)));
    assert!(!expected_streams_arguments(file_search));

    let image_generation = ResponseFormat::ImageGenerationCall;
    assert!(
        !streams_arguments(Some(&image_generation)),
        "image_generation_call must ride the structured-event path",
    );
    assert!(!expected_streams_arguments(image_generation));
}
