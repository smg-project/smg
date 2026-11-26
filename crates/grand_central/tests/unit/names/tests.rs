use super::intern;

#[test]
fn a_name_is_interned_once() {
    let a = intern("plugin-a");
    let b = intern("plugin-a");
    assert!(std::ptr::eq(a, b), "one allocation per distinct name");
    assert_ne!(intern("plugin-b"), a);
}
