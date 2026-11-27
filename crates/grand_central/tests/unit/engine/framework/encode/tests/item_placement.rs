use super::*;

#[test]
fn an_item_is_placed_on_an_encoder_of_the_chosen_prefills_class() {
    let mut bed = Bed::new();
    // Two classes: the prefill of class 2 holds fewer requests and is the
    // pair's; the idle encoder of class 1 is not its encoder.
    bed.prefills = vec![row(0, Role::Prefill, 1, 5), row(1, Role::Prefill, 1, 0)];
    bed.prefill_classes = vec![1, 2];
    bed.encoders = vec![
        row(0, Role::Encode, 0, 0),
        row(1, Role::Encode, 0, 4),
        row(2, Role::Encode, 0, 2),
    ];
    bed.facts = vec![pushing(1), pushing(2), pushing(2)];
    bed.keys = vec![Identity::hashed(11), Identity::hashed(12)];
    assert_eq!(bed.placed(), ((1, 0), vec![2, 2], false));
}

#[test]
fn a_request_without_items_reads_no_encoder_and_places_none() {
    let mut bed = Bed::new();
    bed.keys.clear();
    bed.encoders = vec![unready(row(0, Role::Encode, 0, 0))];
    bed.facts = vec![EncoderFacts::default()];
    bed.prefill_classes = vec![0];
    assert_eq!(bed.placed(), ((0, 0), vec![], false));
}

/// Each shape of the encode pool that declines a request with an item: an
/// empty encode pool beside the bed's prefill, then each of the three
/// columns of another length than its list
/// (`a_column_of_another_length_and_a_binding_of_another_plan_decline`).
fn shapes() -> [fn(&mut Bed); 4] {
    [
        |bed| {
            bed.encoders.clear();
            bed.facts.clear();
            bed.prefill_classes.clear();
        },
        |bed| bed.facts.clear(),
        |bed| bed.prefill_classes.push(1),
        |bed| bed.encode_affinity = vec![AffinityFacts::default(); 3],
    ]
}

/// A request with no item that states `images` images, asked over each of
/// those shapes. Over each it takes the pair entry's pair, and no encoder
/// is placed.
fn a_request_without_items_takes_its_pair(images: u16, decode_images: bool) {
    for (at, shape) in shapes().iter().enumerate() {
        let mut bed = Bed::new();
        bed.keys.clear();
        bed.images = images;
        bed.decode_images = decode_images;
        shape(&mut bed);
        let (decision, placed) = bed.decide();
        assert!(
            matches!(decision, Decision::Pick(_)),
            "shape {at}, {images} images: {decision:?}"
        );
        assert_eq!(decision, bed.pair(&engine()), "shape {at}");
        assert!(placed.encoders().is_empty(), "shape {at}");
    }
}

/// A request with no item is answered whatever its host handed in the
/// encode pool's columns: one that states an image, and one of text alone.
/// The classic router holds such a request to no encoder. It lists the
/// encoders for every request, and each of its tests that refuses a
/// request for want of one, or keeps a pair to the workers one hands to,
/// stands behind the test that the request has items
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2149`,
/// `:2187` and `:2215` on main 384b8fd6b).
#[test]
fn a_request_without_items_is_answered_whatever_the_encode_pools_columns_hold() {
    a_request_without_items_takes_its_pair(1, false);
    a_request_without_items_takes_its_pair(0, false);
}

/// So is one that asks images of its decode: the bed's decode takes them,
/// and the pair is the pair entry's. Over the same shapes a decode that
/// takes none leaves the request no pair, and the shed names the decode's
/// stage: what a request asks of its decode is read whatever the encode
/// pool holds. The classic router refuses that request by its pairs'
/// decodes alone, since a request with no item asks no encoder of a pair
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2187`
/// and `:2202-2206` on main 384b8fd6b).
#[test]
fn a_request_without_items_that_asks_images_of_its_decode_reads_no_encode_column() {
    a_request_without_items_takes_its_pair(1, true);
    for (at, shape) in shapes().iter().enumerate() {
        let mut bed = Bed::new();
        bed.keys.clear();
        bed.decode_images = true;
        bed.decodes = vec![Candidate {
            vision: false,
            ..row(0, Role::Decode, 1, 0)
        }];
        shape(&mut bed);
        assert_eq!(bed.stage(), "decode-vision", "shape {at}");
    }
}

#[test]
fn the_one_key_is_placed_under_a_class_that_pushes_and_under_no_other() {
    let mut bed = Bed::new();
    bed.keys = vec![Identity::hashed(11), Identity::hashed(12)];
    bed.one = Some(Identity::hashed(99));
    assert_eq!(bed.placed(), ((0, 0), vec![0], true));
    // A one key with no form is a one key all the same: the items are
    // placed once, with no affinity.
    bed.one = Some(Identity::NONE);
    assert_eq!(bed.placed(), ((0, 0), vec![0], true));
    bed.one = Some(Identity::hashed(99));
    bed.facts = vec![EncoderFacts {
        class: 1,
        ..EncoderFacts::default()
    }];
    assert_eq!(bed.placed(), ((0, 0), vec![0, 0], false));
    // A class that pushes and a request that carries no one key.
    bed.facts = vec![pushing(1)];
    bed.one = None;
    assert_eq!(bed.placed(), ((0, 0), vec![0, 0], false));
}

#[test]
fn the_admitted_rows_answer_by_their_place_in_the_hosts_pools() {
    let mut bed = Bed::new();
    // The idle prefill's class has no open encoder; the idle decode takes
    // no image and the request asks one of its decode.
    bed.prefills = vec![row(0, Role::Prefill, 1, 0), row(1, Role::Prefill, 1, 9)];
    bed.prefill_classes = vec![1, 2];
    bed.prefill_affinity = vec![AffinityFacts::default(); 2];
    bed.decodes = vec![
        Candidate {
            vision: false,
            ..row(0, Role::Decode, 1, 0)
        },
        row(1, Role::Decode, 1, 9),
    ];
    // Each of the three lists is handed its column, by the pool's places.
    bed.decode_affinity = vec![AffinityFacts::default(); 2];
    bed.encoders = vec![
        unready(row(0, Role::Encode, 0, 0)),
        row(1, Role::Encode, 0, 3),
    ];
    bed.facts = vec![pushing(1), pushing(2)];
    bed.encode_affinity = vec![AffinityFacts::default(); 2];
    bed.decode_images = true;
    assert_eq!(bed.placed(), ((1, 1), vec![1], false));
}

#[test]
fn a_gap_no_retry_closes_is_named_by_membership() {
    // An encoder that takes the request is registered and not ready.
    let mut bed = Bed::new();
    bed.encoders = vec![unready(row(0, Role::Encode, 0, 0))];
    assert_eq!(bed.stage(), "encoders");
    // No encoder is registered.
    bed.encoders.clear();
    bed.facts.clear();
    assert_eq!(bed.stage(), "encoders");

    // The only encoder takes no image.
    let mut bed = Bed::new();
    bed.encoders = vec![Candidate {
        vision: false,
        ..row(0, Role::Encode, 0, 0)
    }];
    assert_eq!(bed.stage(), "encode-vision");
    // Without an image the same encoder serves.
    bed.images = 0;
    assert_eq!(bed.placed(), ((0, 0), vec![0], false));

    // A pushing encoder that states no end.
    let mut bed = Bed::new();
    bed.facts = vec![pushing(0)];
    assert_eq!(bed.stage(), "ec-compatible");
    // An encoder states an end no prefill of a pair states.
    let mut bed = Bed::new();
    bed.prefill_classes = vec![0];
    assert_eq!(bed.stage(), "ec-compatible");
    // Neither states an end. No end is no class: the encoder hands to no
    // prefill, and the prefill no encoder hands to takes none.
    bed.facts = vec![pushing(0)];
    assert_eq!(bed.stage(), "ec-compatible");

    // References, and no encoder that processes them.
    let mut bed = Bed::new();
    bed.references = true;
    assert_eq!(bed.stage(), "encode-media");
    bed.facts = vec![EncoderFacts {
        references: true,
        ..pushing(1)
    }];
    assert_eq!(bed.placed(), ((0, 0), vec![0], false));

    // The decode must take images and none of a served pair does.
    let mut bed = Bed::new();
    bed.decodes = vec![Candidate {
        vision: false,
        ..row(0, Role::Decode, 1, 0)
    }];
    bed.decode_images = true;
    assert_eq!(bed.stage(), "decode-vision");
}
