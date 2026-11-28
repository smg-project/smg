use super::*;

#[test]
fn median_is_the_middle_or_the_mean_of_the_two_middles() {
    assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
    assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
    assert_eq!(median(&[]), None);
}

#[test]
fn python_float_text() {
    assert_eq!(py_float(300.0), "300.0");
    assert_eq!(py_float(0.589), "0.589");
    assert_eq!(py_float(-2.0), "-2.0");
    assert_eq!(round3(0.12345), 0.123);
}
