//! The one prompt admission every wire applies before a request reaches its
//! engine: a prompt carries at least one token and leaves at least one
//! position of the engine's context window for output. Each engine's own
//! frontend applies this bound, and the ZMQ wire bypasses those frontends;
//! the engines differ in what they do with a prompt that fills the window
//! (one refuses it after admission with a reason, one admits it with no
//! output budget left, one needs a concrete output limit it cannot derive).
//! Behind this adapter they answer alike: a refusal before the engine, the
//! caller's error, in the words the first engine's frontend uses.

/// Admit a prompt of `prompt_len` tokens to a context window of `window`
/// tokens, as the engine reported it at the handshake (`0` from an engine
/// that reported none: the upper bound is skipped, the empty check stays).
/// An empty prompt, and one that leaves no position for output, are refused.
///
/// The `Ok` value is the number of output positions the window leaves the
/// prompt (at least one): the default output limit on a wire whose engine
/// needs a concrete one; `u32::MAX` under an unreported window or a window
/// wider than the limit's type.
pub(crate) fn admit_prompt(prompt_len: usize, window: u64) -> Result<u32, String> {
    let prompt_len = prompt_len as u64;
    if prompt_len == 0 {
        return Err("the decoder prompt cannot be empty".to_string());
    }
    if window == 0 {
        return Ok(u32::MAX);
    }
    if prompt_len >= window {
        return Err(format!(
            "the decoder prompt (length {prompt_len}) must be shorter than the maximum \
             model length of {window} to leave room for at least one output token"
        ));
    }
    Ok(u32::try_from(window - prompt_len).unwrap_or(u32::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_prompt_is_refused_under_any_window() {
        for window in [0, 1, 4, 1 << 20] {
            assert_eq!(
                admit_prompt(0, window),
                Err("the decoder prompt cannot be empty".to_string()),
                "window {window}"
            );
        }
    }

    #[test]
    fn the_window_leaves_at_least_one_position_for_output() {
        assert_eq!(admit_prompt(1, 4), Ok(3));
        assert_eq!(admit_prompt(3, 4), Ok(1));
        for prompt_len in [4, 5, 1 << 20] {
            let error = admit_prompt(prompt_len, 4).expect_err("no room for output");
            assert!(error.contains("prompt"), "{error}");
            assert!(error.contains(&format!("length {prompt_len}")), "{error}");
            assert!(error.contains("model length of 4"), "{error}");
        }
    }

    #[test]
    fn the_refusal_names_both_lengths_in_the_frontends_words() {
        assert_eq!(
            admit_prompt(1_049_692, 1_048_576),
            Err(
                "the decoder prompt (length 1049692) must be shorter than the maximum model \
                 length of 1048576 to leave room for at least one output token"
                    .to_string()
            )
        );
    }

    #[test]
    fn an_unreported_window_bounds_nothing_but_emptiness() {
        assert_eq!(admit_prompt(1, 0), Ok(u32::MAX));
        assert_eq!(admit_prompt(1_049_692, 0), Ok(u32::MAX));
        assert!(admit_prompt(0, 0).is_err());
    }

    #[test]
    fn the_remaining_positions_saturate_at_the_limits_type() {
        assert_eq!(admit_prompt(1, 1 << 40), Ok(u32::MAX));
        assert_eq!(
            admit_prompt(1, u64::from(u32::MAX) + 1),
            Ok(u32::MAX),
            "exactly the type's range"
        );
        assert_eq!(admit_prompt(2, u64::from(u32::MAX) + 1), Ok(u32::MAX - 1));
    }
}
