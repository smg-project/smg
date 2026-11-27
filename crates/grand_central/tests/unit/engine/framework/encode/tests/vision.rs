use super::*;

/// References that carry images are attended to by the decode: the pair is
/// picked among the decodes that take images when a served pair has one,
/// and among every decode when none has or the request carries no image.
#[test]
fn references_with_images_take_a_decode_with_vision_when_a_pair_has_one() {
    let blind = |at: u32, in_flight: u32| Candidate {
        vision: false,
        ..row(at, Role::Decode, 1, in_flight)
    };
    let mut bed = Bed::new();
    bed.references = true;
    bed.facts = vec![EncoderFacts {
        references: true,
        ..pushing(1)
    }];
    // The idle decode takes no image, the busy one does.
    bed.decodes = vec![blind(0, 0), row(1, Role::Decode, 1, 9)];
    assert_eq!(bed.placed(), ((0, 1), vec![0], false));
    // Without an image the references ask nothing of the decode.
    bed.images = 0;
    assert_eq!(bed.placed(), ((0, 0), vec![0], false));
    // No decode of the pair takes images: the request is placed all the same.
    bed.images = 1;
    bed.decodes = vec![blind(0, 0), blind(1, 9)];
    assert_eq!(bed.placed(), ((0, 0), vec![0], false));
}

/// A request with no item still asks what it asks of its decode. Images
/// asked of the decode keep it off the encode entry's early answer: the pair
/// is picked among the decodes that take images, and where none takes one
/// the shed names the stage.
#[test]
fn a_request_without_items_that_asks_images_of_its_decode_takes_a_decode_with_vision() {
    let blind = |at: u32, in_flight: u32| Candidate {
        vision: false,
        ..row(at, Role::Decode, 1, in_flight)
    };
    let mut bed = Bed::new();
    bed.keys.clear();
    bed.decode_images = true;
    // The idle decode takes no image, the busy one does.
    bed.decodes = vec![blind(0, 0), row(1, Role::Decode, 1, 9)];
    assert_eq!(bed.placed(), ((0, 1), vec![], false));
    bed.decodes = vec![blind(0, 0), blind(1, 9)];
    assert_eq!(bed.stage(), "decode-vision");
}

/// References that carry images ask a decode with vision of a request
/// with no item too, and nothing of one that carries no image.
#[test]
fn references_with_images_and_no_item_take_a_decode_with_vision() {
    let blind = Candidate {
        vision: false,
        ..row(0, Role::Decode, 1, 0)
    };
    let mut bed = Bed::new();
    bed.keys.clear();
    bed.references = true;
    // The idle decode takes no image, the busy one does.
    bed.decodes = vec![blind, row(1, Role::Decode, 1, 9)];
    assert_eq!(bed.placed(), ((0, 1), vec![], false));
    bed.images = 0;
    assert_eq!(bed.placed(), ((0, 0), vec![], false));
}

/// A video asks vision as an image does, of the encoder that takes the
/// request and, beside references, of its decode. A request whose one item
/// is a video is refused under the vision gap where the only encoder takes
/// no images, and its item goes to the encoder that takes them where one
/// does, busy as it is. References beside a video take a decode that takes
/// images, with an item and with none.
///
/// The classic router asks vision for any pixel item, an image or a video
/// (`is_pixel_modality` and `request_needs_vision`,
/// `model_gateway/src/routers/grpc/multimodal/capability.rs:325-344` on
/// main d6517a4fe), and holds each encoder to it
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:1530`).
#[test]
fn a_video_alone_asks_vision_as_an_image_does() {
    let blind = |at: u32, role: Role, in_flight: u32| Candidate {
        vision: false,
        ..row(at, role, u32::from(role == Role::Decode), in_flight)
    };
    let video = || {
        let mut bed = Bed::new();
        bed.images = 0;
        bed.video = 1;
        bed
    };
    // The only encoder takes no image.
    let mut bed = video();
    bed.encoders = vec![blind(0, Role::Encode, 0)];
    assert_eq!(bed.stage(), "encode-vision");
    // Beside one that takes images the item goes there, busy as it is.
    bed.encoders = vec![blind(0, Role::Encode, 0), row(1, Role::Encode, 0, 9)];
    bed.facts = vec![pushing(1), pushing(1)];
    assert_eq!(bed.placed(), ((0, 0), vec![1], false));

    // References beside a video: the idle decode takes no image, the busy
    // one does.
    let mut bed = video();
    bed.references = true;
    bed.facts = vec![EncoderFacts {
        references: true,
        ..pushing(1)
    }];
    bed.decodes = vec![blind(0, Role::Decode, 0), row(1, Role::Decode, 1, 9)];
    assert_eq!(bed.placed(), ((0, 1), vec![0], false));
    // And with no item.
    bed.keys.clear();
    assert_eq!(bed.placed(), ((0, 1), vec![], false));
}

#[test]
fn a_served_pair_without_vision_ahead_leaves_the_question_of_vision_open() {
    // Two pair classes under no table: the idle prefill's decode takes no
    // image, the busy prefill's does, and both prefills are served. The
    // pass goes on past the first served prefill, so a request that asks
    // an image of its decode is placed on the pair that takes one.
    let mut bed = Bed::new();
    bed.prefills = vec![row(0, Role::Prefill, 1, 0), row(1, Role::Prefill, 2, 9)];
    bed.prefill_classes = vec![1, 1];
    bed.decodes = vec![
        Candidate {
            vision: false,
            ..row(0, Role::Decode, 1, 0)
        },
        row(1, Role::Decode, 2, 9),
    ];
    bed.decode_images = true;
    assert_eq!(bed.placed(), ((1, 1), vec![0], false));
    // References with an image take that pair too.
    bed.decode_images = false;
    bed.references = true;
    bed.facts = vec![EncoderFacts {
        references: true,
        ..pushing(1)
    }];
    assert_eq!(bed.placed(), ((1, 1), vec![0], false));
}
