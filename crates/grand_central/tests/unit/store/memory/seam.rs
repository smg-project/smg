use std::cell::RefCell;

type Hook = Box<dyn FnOnce()>;

thread_local! {
    static TURNED: RefCell<Option<Hook>> = const { RefCell::new(None) };
}

/// `hook` runs once inside this thread's next apply of a generation:
/// the store's engine holds the generation and the store's world still
/// holds the one before. The apply holds both the store's locks there:
/// a hook asks for neither. A controller's cases alone stand there.
#[cfg(feature = "controller")]
pub(crate) fn on_turn(hook: impl FnOnce() + 'static) {
    TURNED.with(|held| *held.borrow_mut() = Some(Box::new(hook)));
}

/// The hook of [`on_turn`] runs. It is taken out first.
pub(super) fn turned() {
    let hook = TURNED.with(|held| held.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}
