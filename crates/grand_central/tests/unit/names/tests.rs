use super::intern;

thread_local! {
    /// How many names this thread asked [`intern`] for: a case reads it
    /// around a call to show that the call takes no name from the table,
    /// and so not the table's lock.
    pub(crate) static ASKED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[test]
fn a_name_is_interned_once() {
    let a = intern("plugin-a");
    let b = intern("plugin-a");
    assert!(std::ptr::eq(a, b), "one allocation per distinct name");
    assert_ne!(intern("plugin-b"), a);
}
