//! Whether a wire request carries the prompt text: never for a text-only request, kept when the
//! request has media.

use super::wire_prompt_text;

#[test]
fn a_text_only_request_sends_no_prompt_text() {
    assert_eq!(wire_prompt_text("<|user|>hello".to_string(), false), "");
}

#[test]
fn a_request_with_media_keeps_its_prompt_text() {
    assert_eq!(
        wire_prompt_text("<|user|><|image|>describe".to_string(), true),
        "<|user|><|image|>describe"
    );
}
