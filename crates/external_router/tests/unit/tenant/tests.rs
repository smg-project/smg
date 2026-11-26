use super::*;

/// The copies of a request's meta share one set of stamps: a clone copies
/// nothing, and a stamp written on one copy after the clone is that
/// copy's own, as it was when every clone was a copy of the map.
#[test]
fn a_cloned_meta_shares_its_stamps_until_one_copy_writes() {
    let meta = RouteRequestMeta::new(TenantKey::new("t")).with_extension(7u32);
    let mut later = meta.clone();
    assert!(Arc::ptr_eq(&meta.extensions, &later.extensions));
    assert_eq!(later.extension::<u32>(), Some(&7));

    later.insert_extension(8u32);
    assert_eq!(meta.extension::<u32>(), Some(&7));
    assert_eq!(later.extension::<u32>(), Some(&8));
    assert!(!Arc::ptr_eq(&meta.extensions, &later.extensions));
}

/// A write on the only copy (the admission re-stamp, before any clone
/// exists) does not copy the map.
#[test]
fn a_write_on_the_only_copy_keeps_its_map() {
    let mut meta = RouteRequestMeta::new(TenantKey::new("t")).with_extension(7u32);
    let before = Arc::as_ptr(&meta.extensions);
    meta.insert_extension(8u32);
    assert_eq!(Arc::as_ptr(&meta.extensions), before);
    assert_eq!(meta.extension::<u32>(), Some(&8));
}
