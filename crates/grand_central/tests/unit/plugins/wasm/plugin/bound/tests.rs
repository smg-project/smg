use super::bit;

#[test]
fn a_mask_reads_its_bits_and_nothing_past_its_end() {
    let mask = [1u64 << 3, 1];
    assert!(bit(&mask, 3));
    assert!(bit(&mask, 64));
    assert!(!bit(&mask, 4));
    assert!(!bit(&mask, 200));
}
