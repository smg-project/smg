/// The marker is replaced as a whole line: a marker the renderer could
/// not replace is refused rather than left in the text.
#[test]
fn a_marker_without_its_newline_is_refused() {
    assert!(render("package a:b;\n{{types}}\n").is_ok());
    for template in [
        "package a:b;\n{{types}}",
        "package a:b;\n{{types}}\r\n",
        "{{types}} \n",
        "package a:b; {{types}}\n",
    ] {
        assert!(render(template).is_err(), "{template:?}");
    }
}

use super::*;

#[test]
fn the_rendering_starts_with_the_interface_and_its_marker_line() {
    let text = render_types();
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some("interface types {"));
    assert_eq!(lines.next(), Some(GENERATED));
    assert!(text.ends_with("}\n"));
}

#[test]
fn a_template_without_the_marker_is_a_drift() {
    assert!(render("package smg:policy@1.0.0;\n").is_err());
    assert!(render("a\n{{types}}\nb\n")
        .unwrap()
        .starts_with("a\ninterface types {\n"));
}
