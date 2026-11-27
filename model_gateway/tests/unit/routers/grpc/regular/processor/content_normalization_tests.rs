use super::normalize_assistant_content;

#[test]
fn whitespace_only_is_none_real_text_kept_verbatim() {
    assert_eq!(normalize_assistant_content("\n\n".to_string()), None);
    assert_eq!(normalize_assistant_content("  \t".to_string()), None);
    assert_eq!(
        normalize_assistant_content("\n\nDone.".to_string()),
        Some("\n\nDone.".to_string())
    );
}
