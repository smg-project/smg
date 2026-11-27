use super::*;

#[test]
fn test_error_codes() {
    assert_eq!(SglErrorCode::Success as i32, 0);
    assert_eq!(SglErrorCode::InvalidArgument as i32, 1);
}
