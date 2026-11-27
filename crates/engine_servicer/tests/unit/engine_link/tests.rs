use super::*;

#[tokio::test]
async fn without_a_directory_the_slot_holds_none() {
    let slot = TokenizerSlot::new();
    load_tokenizer_once(&slot, None).await;
    assert!(matches!(slot.get(), Some(None)));
}

#[tokio::test]
async fn a_directory_that_does_not_load_leaves_none() {
    let dir = tempfile::tempdir().unwrap();
    let slot = TokenizerSlot::new();
    load_tokenizer_once(&slot, Some(dir.path().to_str().unwrap())).await;
    assert!(matches!(slot.get(), Some(None)));
}

#[tokio::test]
async fn a_slot_already_set_is_left_alone() {
    let dir = tempfile::tempdir().unwrap();
    let slot = TokenizerSlot::new();
    let _ = slot.set(None);
    // Nothing is loaded for a set slot (the directory is not read).
    load_tokenizer_once(&slot, Some(dir.path().to_str().unwrap())).await;
    assert!(matches!(slot.get(), Some(None)));
}
