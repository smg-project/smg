use super::*;

/// Each key draws its own sequence, derived from the request's seed: the
/// items of one request over a tie of encoders do not follow one draw to
/// one encoder.
#[test]
fn the_items_of_one_request_spread_over_a_tie_of_encoders() {
    let mut bed = Bed::new();
    bed.encoders = (0..8).map(|at| row(at, Role::Encode, 0, 0)).collect();
    bed.facts = vec![pushing(1); 8];
    bed.keys = (0..6).map(|item| Identity::hashed(100 + item)).collect();
    let (_, encoders, one) = bed.placed();
    assert!(!one);
    assert_eq!(encoders.len(), 6);
    let first = encoders[0];
    assert!(
        encoders.iter().any(|&encoder| encoder != first),
        "six keys over a tie of eight follow one draw: {encoders:?}"
    );
}

/// No item is placed by where its request stands on a lent ring: that
/// place is the request's routing value's, and the ring is the request's
/// own decision's. A key with no text stays on the encode stage's own ring
/// under every such place. The classic gateway keys an item's encoder by
/// the item's content alone
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs`,
/// lines 2308 to 2358 at f55a0a6bf), so one content keeps its encoder
/// under every routing value.
#[test]
fn an_item_keeps_its_encoder_wherever_its_request_stands_on_a_lent_ring() {
    let document = PolicySpec::epd_presets("consistent_hashing", "power_of_two", "power_of_two");
    let engine = document.expect("the presets").validate();
    let engine = engine.expect("the form validates");
    let mut bed = Bed::new();
    bed.encoders = (0..4).map(|at| row(at, Role::Encode, 0, 0)).collect();
    bed.facts = vec![pushing(1); 4];
    // Six keys far apart, as the hashes of six contents are.
    let spread = |item: u64| Identity::hashed((item + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    bed.keys = (0..6).map(spread).collect();
    let (facts, pool) = bed.asked();
    let encoders = |facts: &RequestFacts<'_>| -> Vec<u32> {
        let (_, placed) = bed.decide_asked(&engine, facts, &pool);
        placed.encoders().iter().map(|t| t.position).collect()
    };
    // No ring lent: each item on the encoder its own key reaches.
    let own = encoders(&facts);
    assert_eq!(own.len(), 6);
    let first = own[0];
    assert!(
        own.iter().any(|&encoder| encoder != first),
        "six keys far apart on one encoder of four: {own:?}"
    );
    // A ring lent that holds no encoder: a key placed by the request's
    // place would go to the encoder at that place modulo the four.
    let ring = InOrder(vec!["grpc://prefill-0:9000".to_owned()]);
    for place in 0..4 {
        let places = [place];
        let lent = RequestFacts {
            ring: Some(&ring),
            ring_positions: &places,
            ..facts
        };
        assert_eq!(encoders(&lent), own, "the request stands at {place}");
    }
}

/// The address the encoder at `at` of a case's list is registered at.
fn encoder_at(at: u32) -> String {
    format!("grpc://encoder-{at}:9000")
}

/// `count` ready encoders, each carrying its address's id as a row of a
/// gateway's list does: a lent ring names a worker by its address.
fn addressed(count: u32) -> Vec<Candidate> {
    (0..count)
        .map(|at| Candidate {
            id: WorkerId::from_address(&encoder_at(at)),
            ..row(at, Role::Encode, 0, 0)
        })
        .collect()
}

/// A caller's ring that reaches the encoders at `places` in that order.
fn reaches(places: &[u32]) -> InOrder {
    InOrder(places.iter().map(|&at| encoder_at(at)).collect())
}

/// The texts of eight items, as a host holds the hex of an item's content
/// hash. The cases' ring places a text by its fingerprint, whose last bits
/// read each byte's last bits alone: these eight stand at eight places
/// modulo eight, so they spread over the encoders of every case's ring.
const TEXTS: [&str; 8] = ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"];

/// The `epd` plan whose encode side is the preset `encode`.
fn encoding_by(encode: &str) -> Engine {
    let document = PolicySpec::epd_presets(encode, "power_of_two", "power_of_two");
    let engine = document.expect("the presets").validate();
    engine.expect("the form validates")
}

/// Each placed key's encoder under `engine`, by its row in the encode pool.
fn encoders_of(bed: &Bed, engine: &Engine) -> Vec<u32> {
    let (decision, placed) = bed.decide_with(engine);
    assert!(matches!(decision, Decision::Pick(_)), "{decision:?}");
    placed.encoders().iter().map(|t| t.position).collect()
}

/// An item that carries its text is placed on the ring its request's caller
/// lends for the items: on the first encoder offered to it that the ring's
/// walk reaches from where the item's own text stands. The classic gateway
/// asks its encode policy for each item with the hex of the item's content
/// hash as the routing key, on the model's ring where that policy reads one
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs`,
/// lines 2637 to 2649 and 2722 to 2772 at 6aa129a5c), so a content keeps
/// the encoder that ring gives it. An encoder the ring names that is not
/// offered, one that is not ready or one of another prefill's class, is
/// passed. A key with no text states no place: the stage's own ring keeps
/// it where it stood with no ring lent.
#[test]
fn an_item_goes_where_the_ring_lent_for_the_items_reaches_from_its_own_text() {
    let engine = encoding_by("consistent_hashing");
    let mut bed = Bed::new();
    bed.encoders = addressed(6);
    bed.facts = vec![pushing(1); 6];
    let texts = TEXTS;
    bed.keys = texts.map(Identity::plain).to_vec();
    // No ring lent: each key on the encoder the stage's own ring gives it.
    let own = encoders_of(&bed, &engine);
    let order = [4, 1, 5, 0, 3, 2];
    let ring = reaches(&order);
    // The first of `offered` the ring's walk reaches from `text`'s place.
    let reached = |text: &str, offered: &[u32]| {
        let from = (ring.position(text) % 6) as usize;
        let mut walk = (0..6).map(|step| order[(from + step) % 6]);
        walk.find(|at| offered.contains(at))
            .expect("an encoder offered")
    };
    let walked = |offered: &[u32]| -> Vec<u32> {
        let of_text = |text: &&str| reached(text, offered);
        texts.iter().map(of_text).collect()
    };
    bed.ring = Some(Box::new(reaches(&order)));
    let lent = encoders_of(&bed, &engine);
    assert_eq!(lent, walked(&[0, 1, 2, 3, 4, 5]));
    assert_ne!(lent, own, "the two rings agree on every key");
    // The first key's owner is not ready, and the encoder behind it on the
    // ring hands to another prefill's class: neither is offered, and the
    // key's walk goes past both.
    let at = order.iter().position(|&at| at == lent[0]);
    let at = at.expect("an encoder of the ring");
    let (down, other) = (order[at], order[(at + 1) % 6]);
    bed.encoders[down as usize] = unready(bed.encoders[down as usize]);
    bed.facts[other as usize] = pushing(2);
    let offered: Vec<u32> = (0..6).filter(|at| ![down, other].contains(at)).collect();
    let past = encoders_of(&bed, &engine);
    assert_eq!(past, walked(&offered));
    assert_eq!(past[0], order[(at + 2) % 6], "the first key passes both");
    // The same contents with no text: no place is stated, and each stays on
    // the encoder the stage's own ring gave it.
    let mut bare = Bed::new();
    bare.encoders = addressed(6);
    bare.facts = vec![pushing(1); 6];
    bare.keys = texts
        .map(|text| Identity::hashed(Identity::plain(text).hash))
        .to_vec();
    bare.ring = Some(Box::new(reaches(&order)));
    assert_eq!(encoders_of(&bare, &engine), own);
}

/// A ring lent for the items that reaches no encoder offered holds none of
/// them yet, or is of another membership than the list. Under the plain
/// ring a key then goes to the encoder at its text's place modulo the
/// encoders offered, in their list's order, as the gateway's own ring
/// policy places a key with no ring
/// (`model_gateway/src/policies/consistent_hashing.rs`,
/// `find_by_consistent_hash`): one content keeps one encoder there too.
#[test]
fn a_ring_that_reaches_no_offered_encoder_places_an_item_by_its_texts_place_over_them() {
    let engine = encoding_by("consistent_hashing");
    let mut bed = Bed::new();
    bed.encoders = addressed(5);
    bed.facts = vec![pushing(1); 5];
    // The second encoder is not ready: four are offered.
    bed.encoders[1] = unready(bed.encoders[1]);
    let offered = [0u32, 2, 3, 4];
    let texts = TEXTS;
    bed.keys = texts.map(Identity::plain).to_vec();
    let elsewhere = || InOrder(vec!["grpc://prefill-0:9000".to_owned()]);
    let at = |text: &&str| offered[(elsewhere().position(text) % 4) as usize];
    let by_place: Vec<u32> = texts.iter().map(at).collect();
    assert!(
        offered.iter().all(|encoder| by_place.contains(encoder)),
        "the texts leave an offered encoder out: {by_place:?}"
    );
    bed.ring = Some(Box::new(elsewhere()));
    assert_eq!(encoders_of(&bed, &engine), by_place);
    // A ring that holds no worker answers a key's place too.
    bed.ring = Some(Box::new(InOrder(Vec::new())));
    assert_eq!(encoders_of(&bed, &engine), by_place);
}

/// The one key placed in the items' stead goes where the ring lent for the
/// items reaches from its own text, as an item's key does: the classic
/// gateway places the items of a request without image fan-out by the hex
/// of one digest over them, under a pushing handoff
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs`,
/// lines 2622 to 2635 at 6aa129a5c).
#[test]
fn the_one_key_goes_where_the_lent_ring_reaches_from_its_own_text() {
    let engine = encoding_by("consistent_hashing");
    let mut bed = Bed::new();
    bed.encoders = addressed(6);
    bed.facts = vec![pushing(1); 6];
    bed.keys = TEXTS.map(Identity::plain).to_vec();
    let order = [2, 5, 0, 4, 1, 3];
    let ring = reaches(&order);
    let mut homes = Vec::new();
    for digest in ["d0", "d1", "d2", "d3", "d4", "d5"] {
        bed.one = Some(Identity::plain(digest));
        bed.ring = Some(Box::new(reaches(&order)));
        let (decision, placed) = bed.decide_with(&engine);
        assert!(matches!(decision, Decision::Pick(_)), "{decision:?}");
        assert!(placed.one(), "{digest}");
        let encoders: Vec<u32> = placed.encoders().iter().map(|t| t.position).collect();
        let home = order[(ring.position(digest) % 6) as usize];
        assert_eq!(encoders, [home], "{digest}");
        homes.push(home);
    }
    assert!(
        homes.iter().any(|&home| home != homes[0]),
        "six digests at one encoder of six: {homes:?}"
    );
}

/// A lender that places a key's text and no head: the trait's own answer
/// for a head.
#[derive(Debug)]
struct PlacesNoHead(InOrder);

impl LentRing for PlacesNoHead {
    fn position(&self, key: &str) -> u64 {
        self.0.position(key)
    }

    fn walk(&self, position: u64, take: &mut dyn FnMut(&str) -> bool) {
        self.0.walk(position, take);
    }
}

/// Under the prefix form of the ring the lender is asked where an item's
/// text stands as a head, and never where it stands as a key: a routing
/// value leads its request's prompt there, and the gateway's own policy of
/// that form looks a header's key up as it looks a head up
/// (`model_gateway/src/policies/prefix_hash.rs`, `select_worker_impl`).
/// The crate's test ring answers another number for a head than for a key,
/// so an item placed by the wrong question lands elsewhere. A lender that
/// places no head lends nothing: the stage's own ring places the item, as
/// with no ring lent.
#[test]
fn under_the_prefix_form_an_item_goes_where_its_text_stands_as_a_head() {
    let engine = encoding_by("prefix_hash");
    let mut bed = Bed::new();
    bed.encoders = addressed(6);
    bed.facts = vec![pushing(1); 6];
    let texts = TEXTS;
    bed.keys = texts.map(Identity::plain).to_vec();
    let own = encoders_of(&bed, &engine);
    let order = [3, 0, 4, 2, 5, 1];
    let ring = reaches(&order);
    let by = |place: u64| order[(place % 6) as usize];
    let as_head = |text: &&str| {
        by(ring
            .head_position(Head::Text(text))
            .expect("a head's place"))
    };
    let as_heads: Vec<u32> = texts.iter().map(as_head).collect();
    let as_keys: Vec<u32> = texts.iter().map(|text| by(ring.position(text))).collect();
    assert_ne!(as_heads, as_keys, "the two questions answer alike");
    bed.ring = Some(Box::new(reaches(&order)));
    assert_eq!(encoders_of(&bed, &engine), as_heads);
    bed.ring = Some(Box::new(PlacesNoHead(reaches(&order))));
    assert_eq!(encoders_of(&bed, &engine), own);
}

/// A lender that counts what it is asked: a key's place, a head's place and
/// a walk.
#[derive(Debug)]
struct Counted(InOrder, Arc<[AtomicU32; 3]>);

impl LentRing for Counted {
    fn position(&self, key: &str) -> u64 {
        self.1[0].fetch_add(1, Ordering::Relaxed);
        self.0.position(key)
    }

    fn head_position(&self, head: Head<'_>) -> Option<u64> {
        self.1[1].fetch_add(1, Ordering::Relaxed);
        self.0.head_position(head)
    }

    fn walk(&self, position: u64, take: &mut dyn FnMut(&str) -> bool) {
        self.1[2].fetch_add(1, Ordering::Relaxed);
        self.0.walk(position, take);
    }
}

/// The ring lent for the items is asked once a key that carries its text,
/// by the encode profile's word: a key's place under a stage that places a
/// routing value by its own text, a head's place under the prefix form, and
/// nothing under a profile that reads no ring. A key with no text asks
/// nothing, and neither does a request without items. The classic gateway
/// looks the model's ring up once an item, and reads that ring only where
/// its encode policy reads one
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs`,
/// lines 2722 to 2772 and 2637 to 2649 at 6aa129a5c).
#[test]
fn the_lent_ring_is_asked_once_a_key_and_only_under_a_profile_that_reads_one() {
    let order = [1, 3, 0, 2];
    // What the lender was asked for one request of `keys` under the encode
    // preset `encode`: keys' places, heads' places, walks.
    let asked = |encode: &str, keys: &[Identity<'static>]| {
        let counts: Arc<[AtomicU32; 3]> = Arc::default();
        let mut bed = Bed::new();
        bed.encoders = addressed(4);
        bed.facts = vec![pushing(1); 4];
        bed.keys = keys.to_vec();
        bed.ring = Some(Box::new(Counted(reaches(&order), Arc::clone(&counts))));
        let (decision, _) = bed.decide_with(&encoding_by(encode));
        assert!(
            matches!(decision, Decision::Pick(_)),
            "{encode}: {decision:?}"
        );
        [0, 1, 2].map(|at| counts[at].load(Ordering::Relaxed))
    };
    let mut keys = TEXTS.map(Identity::plain).to_vec();
    keys.push(Identity::hashed(77));
    assert_eq!(asked("consistent_hashing", &keys), [8, 0, 8]);
    assert_eq!(asked("prefix_hash", &keys), [0, 8, 8]);
    assert_eq!(asked("power_of_two", &keys), [0, 0, 0]);
    assert_eq!(asked("consistent_hashing", &[]), [0, 0, 0]);
}
