use super::*;

#[test]
fn op_label_is_bounded() {
    assert_eq!(op_label("pause_generation"), "pause_generation");
    assert_eq!(
        op_label("update_weights_from_disk"),
        "update_weights_from_disk"
    );
    assert_eq!(op_label("inference/v1/generate"), "other");
    assert_eq!(op_label("v1/chat/completions/render"), "other");
    assert_eq!(op_label("../etc"), "other");
    assert_eq!(op_label("sleep"), "sleep");
    assert_eq!(op_label("server_info"), "server_info");
    assert_eq!(op_label(""), "other");
}
