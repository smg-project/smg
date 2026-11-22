//! The cases of the encode handoff's rule and of the derivation of the
//! encode classes. Every name is an opaque token: the rule compares names
//! for equality and reads none.

use super::{hands_to, EncodeClasses, EncodeHandoff, EncodeMember, EncodeOffer};
use crate::engine::EncoderFacts;

/// One worker as a case writes it: its runtime (empty when unknown) and the
/// end it states.
#[derive(Clone, Debug, Default)]
struct Worker {
    runtime: &'static str,
    offer: EncodeOffer,
}

/// A worker of `runtime` that states nothing.
fn silent(runtime: &'static str) -> Worker {
    Worker {
        runtime,
        ..Worker::default()
    }
}

/// An encoder of `runtime` that pushes: a producer on `connector`.
fn pusher(runtime: &'static str, connector: &str) -> Worker {
    silent(runtime)
        .form(EncodeHandoff::Push)
        .producer()
        .on(connector)
}

/// A prefill of `runtime` that takes a push: a consumer on `connector`.
fn taker(runtime: &'static str, connector: &str) -> Worker {
    silent(runtime)
        .form(EncodeHandoff::Push)
        .consumer()
        .on(connector)
}

/// A worker of `runtime` that meets at a rendezvous, as either part.
fn meeter(runtime: &'static str) -> Worker {
    silent(runtime).form(EncodeHandoff::Rendezvous)
}

impl Worker {
    fn form(mut self, handoff: EncodeHandoff) -> Self {
        self.offer.handoff = handoff;
        self
    }

    fn producer(mut self) -> Self {
        self.offer.producer = true;
        self
    }

    fn consumer(mut self) -> Self {
        self.offer.consumer = true;
        self
    }

    fn on(mut self, connector: &str) -> Self {
        self.offer.connector = Some(connector.to_string());
        self
    }

    fn member(&self) -> EncodeMember<'_> {
        EncodeMember {
            runtime: self.runtime,
            offer: &self.offer,
        }
    }
}

fn members(workers: &[Worker]) -> Vec<EncodeMember<'_>> {
    workers.iter().map(Worker::member).collect()
}

/// The classes of `encoders` and `prefills`, each in the order written.
fn derive(encoders: &[Worker], prefills: &[Worker]) -> EncodeClasses {
    EncodeClasses::derive(members(encoders), members(prefills))
}

#[test]
fn an_encoder_and_a_prefill_that_state_one_end_are_one_class() {
    let classes = derive(&[pusher("rt-a", "c1")], &[taker("rt-a", "c1")]);
    assert_eq!(classes.encoders(), [1]);
    assert_eq!(classes.prefills(), [1]);
    assert_eq!(classes.classes(), 1);
    assert_eq!(classes.list(1), [0]);
    assert_eq!(classes.form(1), EncodeHandoff::Push);
}

/// The recorded fleet of two connectors: the gateway offers the first
/// prefill the first encoder and the second prefill the two others.
#[test]
fn under_a_push_the_connector_tells_two_classes_apart() {
    let classes = derive(
        &[
            pusher("rt-a", "c1"),
            pusher("rt-a", "c2"),
            pusher("rt-a", "c2"),
        ],
        &[taker("rt-a", "c1"), taker("rt-a", "c2")],
    );
    assert_eq!(classes.encoders(), [1, 2, 2]);
    assert_eq!(classes.prefills(), [1, 2]);
    assert_eq!(classes.list(1), [0]);
    assert_eq!(classes.list(2), [1, 2]);
    // A prefill's class by its place, and none past the last.
    assert_eq!(classes.prefill_class(1), 2);
    assert_eq!(classes.prefill_class(2), 0);
}

#[test]
fn under_a_push_an_encoder_states_the_producer_role_and_a_prefill_the_consumer_role() {
    // An encoder that states the consumer role alone states no end, and
    // takes no number.
    let classes = derive(
        &[taker("rt-a", "c1"), pusher("rt-a", "c1")],
        &[taker("rt-a", "c1"), pusher("rt-a", "c1")],
    );
    assert_eq!(classes.encoders(), [0, 1]);
    // A prefill that states the producer role alone names no class.
    assert_eq!(classes.prefills(), [1, 0]);
    assert_eq!(classes.list(1), [1]);
    // A worker that states both roles is a producer as an encoder and a
    // consumer as a prefill.
    let both = || pusher("rt-a", "c1").consumer();
    let classes = derive(&[both()], &[both()]);
    assert_eq!(classes.encoders(), [1]);
    assert_eq!(classes.prefills(), [1]);
}

#[test]
fn a_role_on_no_connector_is_no_end() {
    let unnamed = || {
        silent("rt-a")
            .form(EncodeHandoff::Push)
            .producer()
            .consumer()
    };
    let classes = derive(
        &[unnamed(), unnamed().on(""), pusher("rt-a", "c1")],
        &[unnamed(), unnamed().on(""), taker("rt-a", "c1")],
    );
    assert_eq!(classes.encoders(), [0, 0, 1]);
    assert_eq!(classes.prefills(), [0, 0, 1]);
}

#[test]
fn a_rendezvous_is_the_runtime_alone() {
    // Neither a role nor a connector is asked of a worker that meets.
    let classes = derive(
        &[meeter("rt-b"), meeter("rt-b").on("c1")],
        &[meeter("rt-b"), meeter("rt-b").consumer().on("c2")],
    );
    assert_eq!(classes.encoders(), [1, 1]);
    assert_eq!(classes.prefills(), [1, 1]);
    assert_eq!(classes.list(1), [0, 1]);
    assert_eq!(classes.form(1), EncodeHandoff::Rendezvous);
}

#[test]
fn a_worker_that_states_no_form_has_no_end() {
    // A role on a connector says nothing of the form.
    let roles = || silent("rt-a").producer().consumer().on("c1");
    let classes = derive(
        &[silent("rt-a"), roles(), meeter("rt-a")],
        &[silent("rt-a"), roles(), meeter("rt-a")],
    );
    assert_eq!(classes.encoders(), [0, 0, 1]);
    assert_eq!(classes.prefills(), [0, 0, 1]);
    assert_eq!(classes.classes(), 1);
    assert_eq!(classes.list(0), [0u32; 0]);
    assert_eq!(classes.form(0), EncodeHandoff::None);
}

#[test]
fn a_worker_of_an_unknown_runtime_has_no_end() {
    let classes = derive(
        &[meeter(""), pusher("", "c1"), meeter("rt-a")],
        &[meeter(""), taker("", "c1"), meeter("rt-a")],
    );
    assert_eq!(classes.encoders(), [0, 0, 1]);
    assert_eq!(classes.prefills(), [0, 0, 1]);
}

#[test]
fn with_no_encoder_that_states_an_end_there_is_no_class_and_no_table() {
    let encoders = [silent("rt-a"), taker("rt-a", "c1")];
    let prefills = [taker("rt-a", "c1"), meeter("rt-a")];
    // Nothing can name a class, so the prefills are not read.
    let mut read = 0;
    let unread = members(&prefills).into_iter().inspect(|_| read += 1);
    let classes = EncodeClasses::derive(members(&encoders), unread);
    assert_eq!(read, 0);
    assert_eq!(classes, EncodeClasses::default());
    assert_eq!((classes.classes(), classes.bytes()), (0, 0));
    assert!(classes.encoders().is_empty() && classes.prefills().is_empty());
    // The empty value names class 0 at every place.
    assert_eq!(classes.prefill_class(0), 0);
    assert_eq!(classes.prefill_class(1), 0);
    // No encoder at all reads the same.
    assert_eq!(derive(&[], &prefills), EncodeClasses::default());
}

#[test]
fn one_end_on_two_runtimes_is_two_classes() {
    let classes = derive(
        &[meeter("rt-a"), meeter("rt-b")],
        &[meeter("rt-b"), meeter("rt-a"), meeter("rt-c")],
    );
    assert_eq!(classes.encoders(), [1, 2]);
    // A prefill of a runtime no encoder is of names no class.
    assert_eq!(classes.prefills(), [2, 1, 0]);
}

#[test]
fn one_runtime_under_two_forms_is_two_classes() {
    let classes = derive(
        &[pusher("rt-a", "c1"), meeter("rt-a")],
        &[meeter("rt-a"), taker("rt-a", "c1")],
    );
    assert_eq!(classes.encoders(), [1, 2]);
    assert_eq!(classes.prefills(), [2, 1]);
    assert_eq!(classes.form(1), EncodeHandoff::Push);
    assert_eq!(classes.form(2), EncodeHandoff::Rendezvous);
}

#[test]
fn a_class_is_numbered_as_its_first_encoder_comes_and_lists_its_encoders_in_order() {
    let classes = derive(
        &[
            pusher("rt-a", "c2"),
            silent("rt-a"),
            pusher("rt-a", "c1"),
            pusher("rt-a", "c2"),
            meeter("rt-b"),
            pusher("rt-a", "c1"),
        ],
        &[],
    );
    assert_eq!(classes.encoders(), [1, 0, 2, 1, 3, 2]);
    assert_eq!(classes.classes(), 3);
    assert_eq!(classes.list(1), [0, 3]);
    assert_eq!(classes.list(2), [2, 5]);
    assert_eq!(classes.list(3), [4]);
    // No class past the last, and no list.
    assert_eq!(classes.list(4), [0u32; 0]);
    assert_eq!(classes.form(4), EncodeHandoff::None);
    // Four bytes an encoder, four a list entry (an encoder with an end),
    // and a list's end and a form a class.
    assert_eq!(classes.bytes(), 6 * 4 + 5 * 4 + 3 * 4 + 3);
}

/// The rule as the design states it, written out apart from the code under
/// test: one known runtime, one form that is stated, and under a push the
/// producer role on the encoder, the consumer role on the prefill and one
/// connector that has a name.
fn rule(encoder: &Worker, prefill: &Worker) -> bool {
    let (ours, theirs) = (&encoder.offer, &prefill.offer);
    if encoder.runtime.is_empty() || encoder.runtime != prefill.runtime {
        return false;
    }
    match (ours.handoff, theirs.handoff) {
        (EncodeHandoff::Rendezvous, EncodeHandoff::Rendezvous) => true,
        (EncodeHandoff::Push, EncodeHandoff::Push) => {
            ours.producer
                && theirs.consumer
                && ours.connector == theirs.connector
                && ours
                    .connector
                    .as_deref()
                    .is_some_and(|name| !name.is_empty())
        }
        _ => false,
    }
}

#[test]
fn the_classes_hold_exactly_the_handoffs_the_rule_makes_worker_by_worker() {
    // Every offer over three runtimes, the three forms, the two roles and
    // four connectors, once as an encoder and once as a prefill.
    let mut grid = Vec::new();
    for runtime in ["rt-a", "rt-b", ""] {
        for handoff in [
            EncodeHandoff::None,
            EncodeHandoff::Push,
            EncodeHandoff::Rendezvous,
        ] {
            for (producer, consumer) in [(false, false), (true, false), (false, true), (true, true)]
            {
                for connector in [None, Some(""), Some("c1"), Some("c2")] {
                    grid.push(Worker {
                        runtime,
                        offer: EncodeOffer {
                            handoff,
                            producer,
                            consumer,
                            connector: connector.map(str::to_string),
                            references: false,
                        },
                    });
                }
            }
        }
    }
    assert_eq!(grid.len(), 144);
    let classes = derive(&grid, &grid);
    assert_eq!(classes.encoders().len(), grid.len());
    assert_eq!(classes.prefills().len(), grid.len());
    let mut handoffs = 0;
    for (e, &encoder_class) in classes.encoders().iter().enumerate() {
        for (p, &prefill_class) in classes.prefills().iter().enumerate() {
            let by_rule = rule(&grid[e], &grid[p]);
            let by_class = encoder_class != 0 && encoder_class == prefill_class;
            assert_eq!(by_class, by_rule, "encoder {e} and prefill {p}");
            assert_eq!(hands_to(&grid[e].member(), &grid[p].member()), by_rule);
            assert_eq!(classes.prefill_class(p), prefill_class);
            handoffs += usize::from(by_rule);
        }
    }
    // Two runtimes: under a rendezvous sixteen offers a side; under a push a
    // connector has two producers beside two consumers, on two connectors.
    assert_eq!(handoffs, 2 * (16 * 16 + 2 * 2 * 2));
    // A list is its class's encoders, in the order given, and the lists
    // name every encoder with a class once.
    let mut listed = 0;
    for class in 1..=classes.classes() {
        let list = classes.list(class);
        assert!(list.is_sorted(), "class {class}");
        for &at in list {
            assert_eq!(classes.encoders()[at as usize], class);
        }
        listed += list.len();
    }
    let classed = classes.encoders().iter().filter(|&&class| class != 0);
    assert_eq!(listed, classed.count());
    // A class is numbered as its first encoder comes.
    let firsts: Vec<u32> = (1..=classes.classes())
        .map(|class| classes.list(class)[0])
        .collect();
    assert!(firsts.is_sorted());
}

/// What a decision reads of an encoder beside its row comes from its class
/// and its own offer. A pusher that states no role has no class and still
/// pushes. An encoder that processes references says so whatever its class.
/// An offer past the encoders the classes were derived from takes class `0`,
/// and so does every offer where there is no class.
#[test]
fn an_encoders_facts_are_its_class_and_what_its_offer_states() {
    let mut reads = meeter("rt-b");
    reads.offer.references = true;
    let encoders = [
        pusher("rt-a", "c1"),
        silent("rt-a").form(EncodeHandoff::Push).on("c1"),
        reads,
        silent("rt-a"),
    ];
    let classes = derive(&encoders, &[taker("rt-a", "c1"), meeter("rt-b")]);
    assert_eq!(classes.encoders(), [1, 0, 2, 0]);
    let of = |class, pushes, references| EncoderFacts {
        class,
        pushes,
        references,
    };
    let offers = || encoders.iter().map(|worker| &worker.offer);
    let facts: Vec<EncoderFacts> = classes.encoder_facts(offers());
    let stated = [
        of(1, true, false),
        of(0, true, false),
        of(2, false, true),
        of(0, false, false),
    ];
    assert_eq!(facts, stated);

    // One offer more than the classes were derived from.
    let late = pusher("rt-a", "c1");
    let facts: Vec<EncoderFacts> = classes.encoder_facts(offers().chain([&late.offer]));
    assert_eq!(facts[..4], stated);
    assert_eq!(facts[4], of(0, true, false));

    // No class at all: each offer keeps what it states.
    let facts: Vec<EncoderFacts> = EncodeClasses::default().encoder_facts(offers());
    let unclassed = stated.map(|fact| EncoderFacts { class: 0, ..fact });
    assert_eq!(facts, unclassed);
}
