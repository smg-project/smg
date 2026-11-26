//! The cases of a pair whose two sides read the ring its caller lends by
//! two words: one places a routing value by its own text and the other by
//! the request's heads, or the two cut a prompt by two rules. Each side is
//! stated its own places ([`RequestFacts::decode_ring_positions`]).

use super::*;

/// A ring of the addresses it holds in order ([`Lent`]) that says what it
/// was asked: each key by its text and each head by its size. Its two
/// recipes are those of [`Lent`], which give one text two places. With
/// `heads` off it places no head.
#[derive(Debug)]
struct Told {
    ring: Lent,
    heads: bool,
    asked: Mutex<Vec<String>>,
}

impl LentRing for Told {
    fn position(&self, key: &str) -> u64 {
        self.asked.lock().push(format!("key {key}"));
        self.ring.position(key)
    }

    fn head_position(&self, head: Head<'_>) -> Option<u64> {
        self.asked.lock().push(format!("head {head:?}"));
        self.ring.head_position(head).filter(|_| self.heads)
    }

    fn walk(&self, position: u64, take: &mut dyn FnMut(&str) -> bool) {
        self.ring.walk(position, take);
    }
}

/// What a pair's request states of the ring it is lent: whether the ring
/// is lent, the request's own list of places, which is its prefill side's,
/// and its decode side's own.
type Stated = (bool, Vec<u64>, Option<Vec<u64>>);

/// The stock pair of two presets.
fn pair(prefill: &str, decode: &str) -> PolicySpec {
    PolicySpec::pair_presets(prefill, decode).expect("the pair presets")
}

/// The ring's prefix form at a depth of 128 tokens, as a scorer of a
/// profile.
fn prefix_at_128() -> ScoreStage {
    ScoreStage::ConsistentHash(ConsistentHashParams {
        prefix_tokens: Some(128),
        ..ConsistentHashParams::default()
    })
}

/// Two prefix forms of two depths: 256 tokens ahead of 128.
fn two_depths() -> PolicySpec {
    let mut document = pair("prefix_hash", "prefix_hash");
    let decode = document.profiles.get_mut("decode").expect("the side");
    decode.score[0] = prefix_at_128().into();
    document
}

/// What a request with `headers` and a prompt of `tokens` states under
/// `document` on a ring that places heads where `heads` says, and what
/// that ring was asked, in order. A second making states what the first
/// took and asks the lender nothing.
fn stated(
    document: &PolicySpec,
    headers: &dyn HeaderRead,
    tokens: Option<&[u32]>,
    heads: bool,
) -> (Stated, Vec<String>) {
    stated_by_id(document, headers, None, tokens, heads)
}

/// [`stated`] for a request that carries its own id, `request_id`.
fn stated_by_id(
    document: &PolicySpec,
    headers: &dyn HeaderRead,
    request_id: Option<&str>,
    tokens: Option<&[u32]>,
    heads: bool,
) -> (Stated, Vec<String>) {
    let engine = engine_of(document);
    let lists = Lists::of(2, 2);
    let classes = lists.every();
    let ring = Told {
        ring: Lent(Vec::new()),
        heads,
        asked: Mutex::default(),
    };
    let request = PlaceRequest {
        headers,
        request_id,
        prompt: Prompt {
            tokens,
            ..Prompt::NONE
        },
        ring: Some(&ring),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
    };
    let made = Made::new(&engine, &request, &[]);
    let read = || {
        let facts = made.make(engine.standing_spec()).facts;
        let decode = facts.decode_ring_positions.map(<[u64]>::to_vec);
        (facts.ring.is_some(), facts.ring_positions.to_vec(), decode)
    };
    let first = read();
    let asked = ring.asked.lock().clone();
    assert_eq!(read(), first, "a second making");
    assert_eq!(*ring.asked.lock(), asked, "a second making asks nothing");
    (first, asked)
}

/// Each side of a pair whose two sides read the lent ring by two words is
/// stated what a single request of its own profile is stated, and the
/// lender is asked once for each word a side reads. The request's own list
/// is its prefill side's, and its decode side's stands beside it.
///
/// A side of the plain ring is stated where its key's own text stands:
/// the routing value's, or its client's value where it has no other. A
/// side of the prefix form is stated where its heads stand: the routing
/// value where a header handed one, which leads the prompt under every
/// rule, else the prompt's heads as its own rule cuts them, also where
/// the request's own id handed the value. Two rules cut
/// a prompt at two depths, and one key gives them one place, asked once.
/// A side with nothing to place by states nothing, and a request neither
/// side of which has a place is lent no ring.
#[test]
fn each_side_of_a_pair_of_two_ring_words_is_stated_its_own_places() {
    // What the lender answers, by its two recipes.
    let recipes = Lent(Vec::new());
    let text = |key: &str| recipes.position(key);
    let head = |head: Head<'_>| recipes.head_position(head).expect("a head's place");
    let ids: Vec<u32> = (0..300).collect();
    let (deep, shallow) = (Head::Tokens(&ids[..256]), Head::Tokens(&ids[..128]));
    let keyed = [("x-smg-routing-key", "conv-1")];
    let client = [("x-forwarded-for", "10.0.0.1")];
    let (ringed, prefixed) = ("consistent_hashing", "prefix_hash");
    let lent = |prefill: Vec<u64>, decode: Vec<u64>| (true, prefill, Some(decode));
    let told = |asks: &[&str]| asks.iter().map(|ask| (*ask).to_owned()).collect::<Vec<_>>();

    // A routing value: its own text for the plain ring, one head for the
    // prefix form.
    let both = told(&["head Text(6)", "key conv-1"]);
    let (key, led) = (text("conv-1"), head(Head::Text("conv-1")));
    assert_ne!(key, led, "one text, two places");
    let two_forms = stated(&pair(ringed, prefixed), &keyed, Some(&ids), true);
    assert_eq!(two_forms, (lent(vec![key], vec![led]), both.clone()));
    let reversed = stated(&pair(prefixed, ringed), &keyed, Some(&ids), true);
    assert_eq!(reversed, (lent(vec![led], vec![key]), both));

    // A client's value places the plain ring's side alone: the prefix form
    // reads the prompt.
    let both = told(&["key 10.0.0.1", "head Tokens(256)"]);
    let (key, cut) = (text("10.0.0.1"), head(deep));
    let two_forms = stated(&pair(ringed, prefixed), &client, Some(&ids), true);
    assert_eq!(two_forms, (lent(vec![key], vec![cut]), both.clone()));
    let reversed = stated(&pair(prefixed, ringed), &client, Some(&ids), true);
    assert_eq!(reversed, (lent(vec![cut], vec![key]), both));

    // A value the request's own id handed, its turn suffix off, places the
    // plain ring's side by its own text and leads no prompt: the prefix
    // form reads the prompt.
    let by_id = |prefill: &str, decode: &str| {
        let mut document = pair(prefill, decode);
        let header = KeySourceSpec {
            header: Some("x-smg-routing-key".to_owned()),
            ..KeySourceSpec::default()
        };
        let rid = KeySourceSpec {
            rid: true,
            ..KeySourceSpec::default()
        };
        document.bindings[0].routing_key = Some(RoutingKeySpec {
            sources: vec![header, rid],
            strip_lineage: true,
        });
        document
    };
    let (id, both) = (Some("conv-2_t1"), told(&["key conv-2", "head Tokens(256)"]));
    let key = text("conv-2");
    let two_forms = stated_by_id(&by_id(ringed, prefixed), &(), id, Some(&ids), true);
    assert_eq!(two_forms, (lent(vec![key], vec![cut]), both.clone()));
    let reversed = stated_by_id(&by_id(prefixed, ringed), &(), id, Some(&ids), true);
    assert_eq!(reversed, (lent(vec![cut], vec![key]), both));

    // No key at all: the plain ring's side states nothing, and the prefix
    // form's keeps its heads.
    let alone = stated(&pair(ringed, prefixed), &(), Some(&ids), true);
    assert_eq!(
        alone,
        (lent(vec![], vec![cut]), told(&["head Tokens(256)"]))
    );

    // Two depths: each side's own head of the prompt, and one place of a
    // routing value, asked once.
    let both = told(&["head Tokens(256)", "head Tokens(128)"]);
    let depths = stated(&two_depths(), &(), Some(&ids), true);
    assert_eq!(depths, (lent(vec![cut], vec![head(shallow)]), both));
    let depths = stated(&two_depths(), &keyed, Some(&ids), true);
    assert_eq!(
        depths,
        (lent(vec![led], vec![led]), told(&["head Text(6)"]))
    );

    // Nothing to place by on either side: no ring is lent.
    let nothing = stated(&pair(ringed, prefixed), &(), None, true);
    assert_eq!(nothing, ((false, vec![], None), told(&[])));
}

/// A side whose own profile reads the ring by two words is stated
/// nothing, as a single request of that profile is, and the other side is
/// still stated its own. Here one side's profile holds the plain ring
/// beside the prefix form.
#[test]
fn a_side_that_reads_the_ring_by_two_words_itself_is_stated_nothing() {
    let key = Lent(Vec::new()).position("conv-1");
    let keyed = [("x-smg-routing-key", "conv-1")];
    let mixed = |side: &str| {
        let mut document = pair("consistent_hashing", "consistent_hashing");
        let profile = document.profiles.get_mut(side).expect("the side");
        profile.score.push(prefix_at_128().into());
        document
    };
    let asked = vec!["key conv-1".to_owned()];
    let prefill = stated(&mixed("prefill"), &keyed, None, true);
    assert_eq!(prefill, ((true, vec![], Some(vec![key])), asked.clone()));
    let decode = stated(&mixed("decode"), &keyed, None, true);
    assert_eq!(decode, ((true, vec![key], Some(vec![])), asked));
}

/// A lender that places no head leaves the side of the prefix form alone
/// to its stage's own ring: that side states nothing, and the other side
/// keeps its key's place. Under two rules the lender is asked about one
/// head and no more, and the request is lent no ring.
#[test]
fn a_lender_that_places_no_head_leaves_the_other_sides_place() {
    let key = Lent(Vec::new()).position("conv-1");
    let ids: Vec<u32> = (0..300).collect();
    let keyed = [("x-smg-routing-key", "conv-1")];
    let (ringed, prefixed) = ("consistent_hashing", "prefix_hash");
    let both = vec!["head Text(6)".to_owned(), "key conv-1".to_owned()];
    let two_forms = stated(&pair(ringed, prefixed), &keyed, None, false);
    assert_eq!(two_forms, ((true, vec![key], Some(vec![])), both.clone()));
    let reversed = stated(&pair(prefixed, ringed), &keyed, None, false);
    assert_eq!(reversed, ((true, vec![], Some(vec![key])), both));
    let depths = stated(&two_depths(), &(), Some(&ids), false);
    let once = vec!["head Tokens(256)".to_owned()];
    assert_eq!(depths, ((false, vec![], None), once));
}

/// A pair of two ring words is placed from each side's own place on the
/// ring its caller lends: the side of the plain ring on the first worker
/// of its list that the walk reaches from where the key's own text
/// stands, the side of the prefix form from where the key stands as a
/// head, as each of the gateway's two policies walks the pair's one ring
/// (`model_gateway/src/routers/common/placement.rs`, lines 3016 to 3031
/// and 3098 to 3101 at a06805492). The key here is one whose two places
/// give a pair that neither place alone gives. The decode decided again
/// is placed from the decode side's own place too: past the decode it
/// refuses, the next one the walk reaches from there.
#[test]
fn a_pair_of_two_ring_words_is_placed_from_each_sides_own_place() {
    let (ringed, prefixed) = ("consistent_hashing", "prefix_hash");
    for (prefill, decode) in [(ringed, prefixed), (prefixed, ringed)] {
        let mut document = pair(prefill, decode);
        document.plans.pd_pair.decode = DecodePick::AfterPrefill;
        let engine = hosted(engine_of(&document));
        let lists = Lists::of(3, 3);
        let classes = lists.every();
        // The ring names the six workers, the two lists mixed: a prefill
        // or a decode, by its place in its list.
        let order = [
            (false, 1),
            (true, 2),
            (false, 0),
            (true, 0),
            (true, 1),
            (false, 2),
        ];
        let address = |&(prefill, at): &(bool, usize)| {
            let list = if prefill {
                &lists.prefills
            } else {
                &lists.decodes
            };
            list[at].url().to_owned()
        };
        let ring = Lent(order.iter().map(address).collect());
        // A side's workers along the ring from `start`, by their places in
        // that side's list.
        let along = |start: u64, prefill: bool| -> Vec<usize> {
            let walk = order.iter().cycle().skip((start % 6) as usize).take(6);
            let side = walk.filter(|side| side.0 == prefill);
            side.map(|side| side.1).collect()
        };
        // Where each side stands for `key`: the plain ring's side at the
        // key's own text, the prefix form's at the key as a head.
        let places = |key: &str| {
            let text = ring.position(key);
            let head = ring.head_position(Head::Text(key)).expect("a head's place");
            if prefill == ringed {
                (text, head)
            } else {
                (head, text)
            }
        };
        // A key that tells each side's own place from one place for both:
        // at its first decision, and at its decode decided again.
        let telling = |key: &String| {
            let (prefill_at, decode_at) = places(key);
            let own = (along(prefill_at, true)[0], along(decode_at, false)[0]);
            let one = |at: u64| (along(at, true)[0], along(at, false)[0]);
            let next = |at: u64| along(at, false).into_iter().find(|&decode| decode != own.1);
            own != one(prefill_at) && own != one(decode_at) && next(decode_at) != next(prefill_at)
        };
        let key = (0..256).map(|at| format!("conv-{at}")).find(telling);
        let key = key.expect("a key whose two places tell the sides apart");
        let (prefill_at, decode_at) = places(&key);
        let decodes = along(decode_at, false);

        let headers = [("x-smg-routing-key", key.as_str())];
        let refused = Cell::new(None);
        let admits = |worker: &dyn Worker| Some(worker_id(worker)) != refused.get();
        let request = PlaceRequest {
            headers: &headers,
            ring: Some(&ring),
            decode_again: true,
            admits: Some(&admits),
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
        };
        let mut placement = engine.place(&request).expect("a placement");
        let own = (along(prefill_at, true)[0], decodes[0]);
        assert_eq!(lists.places(&placement), own, "{prefill}, {decode}");
        placement.sent(Side::Prefill);
        refused.set(Some(lists.decode_id(decodes[0])));
        assert_eq!(placement.decode_again(&request), Ok(()));
        let again = (own.0, decodes[1]);
        assert_eq!(lists.places(&placement), again, "{prefill}, {decode}");
    }
}
