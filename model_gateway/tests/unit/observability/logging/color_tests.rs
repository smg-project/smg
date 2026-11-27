use std::ffi::OsStr;

use super::{color_for, set, truthy};

#[test]
fn a_terminal_is_colored_unless_refused_and_a_pipe_only_when_forced() {
    // (terminal, refused, forced)
    assert!(color_for(true, false, false));
    assert!(!color_for(true, true, false));
    assert!(color_for(true, true, true));
    assert!(!color_for(false, false, false));
    assert!(!color_for(false, true, false));
    assert!(color_for(false, false, true));
    assert!(color_for(false, true, true));
}

/// A force variable set to `0` does not force (clicolors); `NO_COLOR` set
/// to anything non-empty, `0` included, refuses (no-color.org); an empty
/// value is unset for both. The readers are pure: the process environment
/// is not touched (the lib tests run in parallel and read it).
#[test]
fn a_force_variable_set_to_zero_does_not_force_and_an_empty_one_is_unset() {
    let zero = Some(OsStr::new("0"));
    let empty = Some(OsStr::new(""));
    let one = Some(OsStr::new("1"));
    assert!(set(zero));
    assert!(!truthy(zero));
    assert!(!set(empty));
    assert!(!truthy(empty));
    assert!(set(one));
    assert!(truthy(one));
    assert!(!set(None));
    assert!(!truthy(None));
}
