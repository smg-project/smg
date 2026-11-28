use super::*;

/// Whether the classes pair prefill `prefill` with decode `decode`.
fn pairs(classes: &PartnerClasses, prefill: usize, decode: usize) -> bool {
    classes.pairs(prefill, decode)
}

/// The classes of `partners` say the pairs `partners` names, and no
/// other, under a table the engine honours whole.
fn says(decodes: usize, partners: &[&[u32]]) -> PartnerClasses {
    let classes = PartnerClasses::number(decodes, partners.iter().copied());
    assert_eq!(classes.table().validate(), Ok(()));
    assert_eq!(classes.prefills().len(), partners.len());
    assert_eq!(classes.decodes().len(), decodes);
    for (prefill, list) in partners.iter().enumerate() {
        for decode in 0..decodes {
            assert_eq!(
                pairs(&classes, prefill, decode),
                list.contains(&(decode as u32)),
                "prefill {prefill} with decode {decode}"
            );
        }
    }
    classes
}

#[test]
fn a_fleet_whose_workers_all_pair_is_one_class_a_side() {
    let every: &[u32] = &[0, 1, 2];
    let classes = says(3, &[every, every]);
    assert_eq!(classes.prefills(), [1, 1]);
    assert_eq!(classes.decodes(), [1, 1, 1]);
    assert_eq!(classes.table().of(&1), [1]);
}

#[test]
fn two_groups_that_never_pair_are_two_classes_a_side() {
    let classes = says(4, &[&[0, 1], &[2, 3], &[0, 1]]);
    assert_eq!(classes.prefills(), [1, 2, 1]);
    assert_eq!(classes.decodes(), [1, 1, 2, 2]);
    assert_eq!(classes.table().of(&1), [1]);
    assert_eq!(classes.table().of(&2), [2]);
}

/// A decode two prefill classes name is a class of its own, and both
/// lists name it: a worker stands once in its list.
#[test]
fn a_decode_two_prefill_classes_share_is_named_by_both_lists() {
    let classes = says(3, &[&[0, 1], &[1, 2]]);
    assert_eq!(classes.prefills(), [1, 2]);
    assert_eq!(classes.decodes(), [1, 2, 3]);
    assert_eq!(classes.table().of(&1), [1, 2]);
    assert_eq!(classes.table().of(&2), [2, 3]);
}

#[test]
fn a_worker_nobody_pairs_with_is_class_zero() {
    let classes = says(3, &[&[], &[2]]);
    assert_eq!(classes.prefills(), [0, 1]);
    assert_eq!(classes.decodes(), [0, 0, 1]);
}

#[test]
fn a_place_past_the_decode_list_names_no_decode() {
    let classes = PartnerClasses::number(2, [&[1u32, 7][..], &[9][..]]);
    assert_eq!(classes.table().validate(), Ok(()));
    assert_eq!(classes.prefills(), [1, 0]);
    assert_eq!(classes.decodes(), [0, 1]);
    assert!(pairs(&classes, 0, 1));
}

/// No prefill and no decode: no class, and no table. A decision over
/// such classes pairs nothing, since class `0` pairs with nobody.
#[test]
fn empty_lists_have_no_class_and_no_table() {
    let classes = says(0, &[]);
    assert!(classes.table().is_empty());
    let unpaired = says(2, &[&[], &[]]);
    assert_eq!(unpaired.prefills(), [0, 0]);
    assert_eq!(unpaired.decodes(), [0, 0]);
    assert!(unpaired.table().is_empty());
}

/// A relation with no pattern to it: every prefill's list is kept.
#[test]
fn the_classes_say_the_pairs_they_were_numbered_from() {
    let lists: Vec<Vec<u32>> = (0u32..12)
        .map(|prefill| {
            (0u32..9)
                .filter(|decode| (prefill * 7 + decode * 5 + prefill * decode) % 3 == 0)
                .collect()
        })
        .collect();
    let partners: Vec<&[u32]> = lists.iter().map(Vec::as_slice).collect();
    says(9, &partners);
}

/// The classes of `partners`, each prefill's partners and the ones it
/// is paired with first, say both relations and no other, and the
/// first partners stand within the table.
fn says_first(decodes: usize, partners: &[(&[u32], &[u32])]) -> PartnerClasses {
    let classes = PartnerClasses::number_first(decodes, partners.iter().copied());
    assert_eq!(classes.table().validate(), Ok(()));
    assert_eq!(classes.first().within(&classes.table()), Ok(()));
    for (prefill, (list, firsts)) in partners.iter().enumerate() {
        for decode in 0..decodes {
            let (place, at) = (decode as u32, (prefill, decode));
            let paired = list.contains(&place);
            assert_eq!(pairs(&classes, prefill, decode), paired, "{at:?}");
            let goes_first = paired && firsts.contains(&place);
            assert_eq!(classes.pairs_first(prefill, decode), goes_first, "{at:?}");
        }
    }
    classes
}

/// The decodes a prefill class is paired with first stand apart from
/// its other partners: a class's first partners are whole classes of
/// its table.
#[test]
fn the_partners_a_class_takes_first_are_classes_of_their_own() {
    let classes = says_first(3, &[(&[0, 1, 2], &[0, 1])]);
    assert_eq!(classes.prefills(), [1]);
    assert_eq!(classes.decodes(), [1, 1, 2]);
    assert_eq!(classes.table().of(&1), [1, 2]);
    assert_eq!(classes.first().of(1), [1]);
}

/// Two prefills with the same partners are two classes where one of
/// them alone pairs with some of those first.
#[test]
fn a_prefill_with_first_partners_stands_apart_from_one_with_none() {
    let both: &[u32] = &[0, 1];
    let classes = says_first(2, &[(both, &[0]), (both, &[])]);
    assert_eq!(classes.prefills(), [1, 2]);
    assert_eq!(classes.decodes(), [1, 2]);
    assert_eq!(classes.first().of(1), [1]);
    assert_eq!(classes.first().of(2), [0u32; 0]);
}

/// Every partner first is no partner first: the classes are those of
/// the partners alone, and a decision reads no first partner.
#[test]
fn a_fleet_whose_every_partner_goes_first_keeps_no_first_partner() {
    let (near, far): (&[u32], &[u32]) = (&[0, 1], &[2]);
    let lists = [(near, near), (far, far), (near, near)];
    let classes = PartnerClasses::number_first(3, lists);
    assert_eq!(classes, PartnerClasses::number(3, [near, far, near]));
    assert_eq!(classes.first().within(&classes.table()), Ok(()));
    assert!((0..=2).all(|class| classes.first().of(class).is_empty()));
}

/// A first partner that is no partner, or stands past the decode
/// list, is passed over.
#[test]
fn a_first_partner_outside_the_partners_is_passed_over() {
    let classes = PartnerClasses::number_first(3, [(&[0u32, 2][..], &[1u32, 2, 9][..])]);
    assert_eq!(classes.first().within(&classes.table()), Ok(()));
    assert_eq!(classes.decodes(), [1, 0, 2]);
    assert_eq!(classes.table().of(&1), [1, 2]);
    assert_eq!(classes.first().of(1), [2]);
}

/// Two relations with no pattern to them: both are kept, and the first
/// partners stand within the table by the way they are numbered.
#[test]
fn the_classes_say_the_first_partners_they_were_numbered_from() {
    let pairs = |prefill: u32, decode: u32| (prefill * 7 + decode * 5 + prefill * decode) % 3 != 1;
    let goes_first =
        |prefill: u32, decode: u32| (prefill + 2 * decode + prefill * decode).is_multiple_of(4);
    let lists: Vec<[Vec<u32>; 2]> = (0u32..12)
        .map(|prefill| {
            let partners = (0u32..9).filter(|&decode| pairs(prefill, decode));
            let firsts = (0u32..9).filter(|&decode| goes_first(prefill, decode));
            [partners.collect(), firsts.collect()]
        })
        .collect();
    let partners: Vec<(&[u32], &[u32])> = lists
        .iter()
        .map(|[partners, firsts]| (partners.as_slice(), firsts.as_slice()))
        .collect();
    says_first(9, &partners);
}
