use super::py_g;

#[test]
fn general_format_matches_python() {
    assert_eq!(py_g(300.5, 4), "300.5");
    assert_eq!(py_g(300.0, 4), "300");
    assert_eq!(py_g(1.55, 4), "1.55");
    assert_eq!(py_g(500.0, 6), "500");
    assert_eq!(py_g(1000.5, 6), "1000.5");
    assert_eq!(py_g(0.0001234, 4), "0.0001234");
    assert_eq!(py_g(0.00001234, 4), "1.234e-05");
    assert_eq!(py_g(123456.0, 4), "1.235e+05");
    assert_eq!(py_g(0.0, 4), "0");
    assert_eq!(py_g(-2.5, 4), "-2.5");
}
