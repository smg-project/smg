use super::*;

/// An encoder dialed on no wire the request's host states is left out, and
/// the item goes to one that is: here the idle encoder is dialed over gRPC
/// and the busy one over HTTP. A host that states no wire leaves none out,
/// and one that states both takes the idle one.
///
/// The classic router tests an encoder's wire with the rest of what the
/// request asks of it, and keeps the encoders that pass
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2138`
/// and `:2147` on main 384b8fd6b).
#[test]
fn an_encoder_dialed_on_no_wire_the_request_states_is_left_out() {
    let mut bed = Bed::new();
    bed.encoders = vec![
        dialed_on(WireModes::GRPC, 0, 0),
        dialed_on(WireModes::HTTP, 1, 4),
    ];
    bed.facts = vec![pushing(1), pushing(1)];
    assert_eq!(bed.placed(), ((0, 0), vec![0], false));
    bed.wires = WireModes::HTTP;
    assert_eq!(bed.placed(), ((0, 0), vec![1], false));
    bed.wires = WireModes::HTTP.union(WireModes::GRPC);
    assert_eq!(bed.placed(), ((0, 0), vec![0], false));
    // An encoder that is not ready is no encoder to send to, on its wire
    // or off it: the shed is the encode side's, by state.
    bed.wires = WireModes::HTTP;
    bed.encoders[1] = unready(bed.encoders[1]);
    assert_eq!(bed.stage(), "encoders");
}

/// An encoder dialed on two wires takes a request held to one of them: a
/// row states the set of its wires, and one stated wire in the set is
/// enough. The idle encoder is dialed over HTTP and gRPC, and the busy one
/// over gRPC alone: a request held to gRPC takes the idle one, and so does
/// a request held to HTTP, where it is the only one left.
///
/// The classic router compares the request's wire with the one wire its
/// worker is dialed on
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2138`
/// on main 384b8fd6b, by
/// `model_gateway/src/routers/common/placement.rs:88-93`): it holds no
/// encoder on two.
#[test]
fn an_encoder_dialed_on_two_wires_takes_a_request_held_to_one_of_them() {
    let mut bed = Bed::new();
    bed.encoders = vec![
        dialed_on(WireModes::HTTP.union(WireModes::GRPC), 0, 0),
        dialed_on(WireModes::GRPC, 1, 4),
    ];
    bed.facts = vec![pushing(1), pushing(1)];
    bed.wires = WireModes::GRPC;
    assert_eq!(bed.placed(), ((0, 0), vec![0], false));
    bed.wires = WireModes::HTTP;
    assert_eq!(bed.placed(), ((0, 0), vec![0], false));
}

/// A prefill whose class holds no encoder on the request's wires is no
/// candidate of the pair. The idle prefill's one encoder is dialed over
/// gRPC, and the busy prefill's over HTTP: a request held to HTTP takes
/// the busy prefill and its encoder.
///
/// The classic router keeps a pair to the workers an encoder it kept hands
/// to
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2213-2216`
/// on main 384b8fd6b, over the encoders kept at `:2147`).
#[test]
fn a_prefill_whose_encoders_are_on_other_wires_is_not_the_pairs() {
    let mut bed = Bed::new();
    bed.prefills = vec![row(0, Role::Prefill, 1, 0), row(1, Role::Prefill, 1, 5)];
    bed.prefill_classes = vec![1, 2];
    bed.encoders = vec![
        dialed_on(WireModes::GRPC, 0, 0),
        dialed_on(WireModes::HTTP, 1, 0),
    ];
    bed.facts = vec![pushing(1), pushing(2)];
    assert_eq!(bed.placed(), ((0, 0), vec![0], false));
    bed.wires = WireModes::HTTP;
    assert_eq!(bed.placed(), ((1, 0), vec![1], false));
}

/// A gap is named by the encoders on the request's wires. The encoder
/// dialed over HTTP processes no references, and the one over gRPC
/// processes them and takes images: for a request of references with an
/// image held to HTTP, no encoder takes the media. With no wire stated the
/// second encoder takes the request. Where the encoder over gRPC pushes,
/// the gap is the one of an encoder that lacks its role, on whatever wire
/// that encoder is dialed.
///
/// The classic router names a gap by the encoders that pass its test of
/// the wire: with none that would take the request but for its image, the
/// request's media are not supported
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2153-2164`
/// and `:2171-2172` on main 384b8fd6b). Its test of a pushing encoder
/// stands between the two and reads no wire (`:2165-2170`).
#[test]
fn a_gap_is_named_by_the_encoders_on_the_requests_wires() {
    let processes = |references: bool| EncoderFacts {
        class: 1,
        pushes: false,
        references,
    };
    let mut bed = Bed::new();
    bed.encoders = vec![
        dialed_on(WireModes::HTTP, 0, 0),
        dialed_on(WireModes::GRPC, 1, 0),
    ];
    bed.facts = vec![processes(false), processes(true)];
    bed.references = true;
    assert_eq!(bed.placed(), ((0, 0), vec![1], false));
    bed.wires = WireModes::HTTP;
    assert_eq!(bed.stage(), "encode-media");
    bed.facts[1] = EncoderFacts {
        references: true,
        ..pushing(1)
    };
    assert_eq!(bed.stage(), "ec-compatible");
}

/// The pass keeps the hard demand its encode side states in its one test
/// of an encoder, against what each encoder states beside its row
/// ([`EncodePool::demands`]): the runtime as a wire, and vision with the
/// request's pixels. An encoder that states nothing meets no demand for a
/// runtime, as every encoder does under an empty column. A column of
/// another length than the encoders is declined for a request that reads
/// it, and is not asked of one that reads none of it. A window and a way
/// of deciding a constraint are no encoder's to meet.
///
/// The classic router holds an encoder to the request's wire in its one
/// test of an encoder
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2576`
/// on main ccc411cf0), and answers a pushing encoder that takes the media
/// off that wire as one that lacks its role (`:2603-2608`).
#[test]
fn an_encoder_is_held_to_the_encode_sides_demand_by_what_it_states() {
    let on = |runtime: u32| DemandFacts {
        runtime,
        ..DemandFacts::default()
    };
    let mut bed = Bed::new();
    bed.encoders = vec![row(0, Role::Encode, 0, 0), row(1, Role::Encode, 0, 4)];
    bed.facts = vec![pushing(1), pushing(1)];
    bed.demands = Demands {
        runtime: 7,
        ..Demands::default()
    };
    // No column: no encoder states a runtime, and a pushing encoder that
    // takes the media stands off the request's.
    assert_eq!(bed.stage(), "ec-compatible");
    // The first states another runtime, and the second states nothing.
    bed.stated = vec![on(8), DemandFacts::default()];
    assert_eq!(bed.stage(), "ec-compatible");
    // A column of another length than the encoders is its host's mistake,
    // one entry short or one too many: the request is declined.
    bed.stated = vec![on(7)];
    assert_eq!(bed.stage(), "encode-column");
    bed.stated = vec![on(7); 3];
    assert_eq!(bed.stage(), "encode-column");
    // The second states the request's: the item goes there, past the idle
    // first.
    bed.stated = vec![on(8), on(7)];
    assert_eq!(bed.placed(), ((0, 0), vec![1], false));
    // A window and a way of deciding a constraint ask nothing of it.
    bed.demands = Demands {
        input_window: 9,
        output_window: 9,
        decoding: 3,
        ..bed.demands
    };
    assert_eq!(bed.placed(), ((0, 0), vec![1], false));
    // A demand for vision is the test of the request's pixels, and reads
    // no entry: a column of another length is not asked of it. The request
    // holds no image and demands vision, and the idle encoder takes none.
    bed.stated = vec![on(8)];
    bed.images = 0;
    bed.demands = Demands {
        vision: true,
        ..Demands::default()
    };
    bed.encoders[0].vision = false;
    assert_eq!(bed.placed(), ((0, 0), vec![1], false));
    bed.encoders[1].vision = false;
    assert_eq!(bed.stage(), "encode-vision");
}
