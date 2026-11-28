use super::*;

#[test]
fn before_hook_result_default_is_continue_with_empty_extra() {
    let result = BeforeHookResult::default();
    match result {
        BeforeHookResult::Continue(extra) => assert!(extra.is_empty()),
        BeforeHookResult::Reject(_) => panic!("expected Continue"),
    }
}

#[test]
fn storage_operation_is_copy() {
    let op = StorageOperation::CreateConversation;
    let op2 = op; // Copy
    assert_eq!(op, op2);
}

#[test]
fn hook_error_display() {
    let err = HookError::Rejected("bad input".to_string());
    assert_eq!(err.to_string(), "hook rejected: bad input");

    let err = HookError::Internal("timeout".to_string());
    assert_eq!(err.to_string(), "hook error: timeout");
}
