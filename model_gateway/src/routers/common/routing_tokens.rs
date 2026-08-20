//! Routing-token boundary truncation shared by the HTTP, PD and gRPC
//! selection paths.

use crate::observability::metrics::Metrics;

/// Slice `tokens` at the first configured boundary id. Content past a
/// boundary is never shareable across conversations, and match ratios over
/// the full sequence shrink as conversations grow.
pub(crate) fn truncate_slice<'a>(
    tokens: &'a [u32],
    boundaries: &[u32],
    router_type: &'static str,
) -> &'a [u32] {
    if boundaries.is_empty() {
        return tokens;
    }
    match tokens.iter().position(|id| boundaries.contains(id)) {
        Some(cut) => {
            Metrics::record_routing_tokens_truncated(router_type);
            &tokens[..cut]
        }
        None => tokens,
    }
}

/// Owned variant: truncates in place, no reallocation.
pub(crate) fn truncate_owned(
    mut tokens: Vec<u32>,
    boundaries: &[u32],
    router_type: &'static str,
) -> Vec<u32> {
    let keep = truncate_slice(&tokens, boundaries, router_type).len();
    tokens.truncate(keep);
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuts_at_first_of_several_boundaries() {
        assert_eq!(
            truncate_slice(&[1, 2, 900, 3, 901], &[900, 901], "http"),
            &[1, 2]
        );
        assert_eq!(
            truncate_owned(vec![1, 2, 901, 3, 900], &[900, 901], "http"),
            vec![1, 2]
        );
    }

    #[test]
    fn boundary_first_yields_empty() {
        assert_eq!(truncate_slice(&[900, 1], &[900], "http"), &[] as &[u32]);
        assert_eq!(
            truncate_owned(vec![900, 1], &[900], "http"),
            Vec::<u32>::new()
        );
    }

    #[test]
    fn no_boundaries_or_no_match_is_identity() {
        assert_eq!(truncate_slice(&[1, 900, 2], &[], "http"), &[1, 900, 2]);
        assert_eq!(truncate_slice(&[1, 2, 3], &[900], "http"), &[1, 2, 3]);
        assert_eq!(truncate_owned(vec![1, 2], &[900], "http"), vec![1, 2]);
    }
}
