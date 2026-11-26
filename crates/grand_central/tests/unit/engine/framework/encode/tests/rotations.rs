use super::*;

/// The `epd` plan with every side turning its pool's rotation.
fn turning() -> Engine {
    let spec = PolicySpec::epd_presets("round_robin", "round_robin", "round_robin");
    spec.expect("the presets")
        .validate()
        .expect("the form validates")
}

/// The sizes a list is turned at: a small one, and the first at which a
/// single decision draws before it reads its pool.
const TURNED: [u32; 2] = [2, framework::DRAW_FROM as u32];

/// The encoders offered under two classes are two lists of one pool, and
/// each turns a rotation of its own, also where they have one size: a pair
/// that alternates between the classes reaches every encoder of each. A
/// key's run is a whole pass, which turns a rotation by the rows it keeps.
#[test]
fn the_encoders_offered_under_two_classes_turn_rotations_apart() {
    for size in TURNED {
        let engine = turning();
        // Two encode classes of `size` encoders each and a prefill of each,
        // one item a request.
        let mut bed = Bed::new();
        bed.prefills = vec![row(0, Role::Prefill, 1, 0), row(1, Role::Prefill, 1, 0)];
        bed.prefill_classes = vec![1, 2];
        bed.encoders = (0..2 * size)
            .map(|at| row(at, Role::Encode, 0, 0))
            .collect();
        bed.facts = (0..2 * size).map(|at| pushing(1 + at / size)).collect();
        let mut reached = vec![0; bed.encoders.len()];
        for _ in 0..4 * size {
            let (decision, placed) = bed.decide_with(&engine);
            assert!(decision.pick().is_some(), "a pick, not {decision:?}");
            reached[placed.encoders()[0].position as usize] += 1;
        }
        assert!(
            reached.iter().all(|&turns| turns == 2),
            "{size}: {reached:?}"
        );
    }
}

/// Two cuts of the prefills are two lists of one pool: two kinds of media,
/// each served by a class of its own, alternate, and each kind's requests
/// reach every prefill of its class. A side's run is a whole pass too.
#[test]
fn two_cuts_of_the_prefills_turn_rotations_apart() {
    for size in TURNED {
        let engine = turning();
        // The first encoder takes images and the second references.
        let kind = |references: bool| {
            let mut bed = Bed::new();
            bed.prefills = (0..2 * size)
                .map(|at| row(at, Role::Prefill, 1, 0))
                .collect();
            bed.prefill_classes = (0..2 * size).map(|at| 1 + at / size).collect();
            bed.encoders = vec![row(0, Role::Encode, 0, 0), row(1, Role::Encode, 0, 0)];
            bed.encoders[1].vision = false;
            bed.facts = vec![pushing(1), pushing(2)];
            bed.facts[1].references = true;
            bed.references = references;
            bed.images = u16::from(!references);
            bed
        };
        let kinds = [kind(false), kind(true)];
        let mut reached = vec![0; kinds[0].prefills.len()];
        for bed in kinds.iter().cycle().take(4 * size as usize) {
            let (decision, _) = bed.decide_with(&engine);
            reached[pair_of(&decision).0 as usize] += 1;
        }
        assert!(
            reached.iter().all(|&turns| turns == 2),
            "{size}: {reached:?}"
        );
    }
}
