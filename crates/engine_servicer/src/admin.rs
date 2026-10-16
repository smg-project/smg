//! What the admin RPCs check before an engine is asked anything, the same
//! for every servicer that serves them.

use tonic::Status;

/// `FlushCache`'s `timeout_s`, as the Python servicer accepts it: finite and
/// non-negative (a zero flushes at once or refuses), anything else refused
/// as INVALID_ARGUMENT.
pub(crate) fn flush_timeout(timeout_s: f32) -> Result<f32, Status> {
    if !timeout_s.is_finite() || timeout_s < 0.0 {
        return Err(Status::invalid_argument(
            "timeout_s must be finite and non-negative",
        ));
    }
    Ok(timeout_s)
}

#[cfg(test)]
mod tests {
    use tonic::Code;

    use super::*;

    #[test]
    fn a_flush_timeout_is_finite_and_non_negative() {
        assert_eq!(flush_timeout(0.0).unwrap(), 0.0);
        assert_eq!(flush_timeout(2.5).unwrap(), 2.5);
        for refused in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let status = flush_timeout(refused).expect_err("refused");
            assert_eq!(status.code(), Code::InvalidArgument, "{refused}");
            assert_eq!(
                status.message(),
                "timeout_s must be finite and non-negative"
            );
        }
    }
}
