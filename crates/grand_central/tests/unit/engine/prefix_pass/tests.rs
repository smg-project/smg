use super::*;

/// A take empties the slot whichever stage wrote it, and answers only
/// the stage that wrote.
#[test]
fn a_take_clears_whoever_wrote_and_answers_only_the_writer() {
    let (writer, other) = (PrefixStageId::next(), PrefixStageId::next());
    forget_prefix_pass();
    keep_prefix_pass(writer, 3, 1.5);
    // Another stage's take reads nothing, and the slot is empty behind it.
    assert_eq!(take_prefix_pass(other), None);
    assert_eq!(take_prefix_pass(writer), None);
    // The writer's own take answers what it kept, once.
    keep_prefix_pass(writer, 3, 1.5);
    assert_eq!(take_prefix_pass(writer), Some((3, 1.5)));
    assert_eq!(take_prefix_pass(writer), None);
}

/// The first stage to keep holds the slot: a later stage neither writes
/// over it nor withdraws it, and the stage's own withdrawal empties it.
#[test]
fn the_first_stage_to_keep_holds_the_slot() {
    let (first, later) = (PrefixStageId::next(), PrefixStageId::next());
    forget_prefix_pass();
    keep_prefix_pass(first, 2, 0.5);
    keep_prefix_pass(later, 9, 9.0);
    withdraw_prefix_pass(later);
    assert_eq!(take_prefix_pass(first), Some((2, 0.5)));
    keep_prefix_pass(first, 2, 0.5);
    withdraw_prefix_pass(first);
    assert_eq!(take_prefix_pass(first), None);
}

/// No two stages are handed one number, and none is handed zero.
#[test]
fn a_new_stage_takes_a_number_of_its_own() {
    let (one, two) = (PrefixStageId::next(), PrefixStageId::next());
    assert_ne!(one, two);
    assert_ne!(one.0, 0);
    assert_ne!(two.0, 0);
}
