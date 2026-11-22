//! The room a membership write takes for a pool's list.

use super::super::list_of;

#[test]
fn a_standing_pools_list_takes_one_member_more_without_growing() {
    for standing in [1_usize, 7, 512] {
        let mut list = list_of::<u64>(standing);
        let room = list.capacity();
        list.extend(0..=standing as u64);
        assert_eq!(list.capacity(), room, "{standing} standing: the list grew");
    }
}

#[test]
fn a_pool_that_did_not_stand_takes_no_room() {
    assert_eq!(list_of::<u64>(0).capacity(), 0);
}
