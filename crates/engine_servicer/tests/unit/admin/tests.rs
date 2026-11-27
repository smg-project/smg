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
