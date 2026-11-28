//! Cases of the overload settings: the veto's switch, its shed word and
//! its two thresholds, in the settings' own document and beside a given
//! one.

use super::*;

/// A threshold at none switches its signal off in the gateway, on its
/// worker objects, and no document carries a threshold. With the veto
/// on, one signal off leaves the block switched on, whatever the other
/// is: the settings mean the preset, with nothing written over it. Both
/// off is the block switched off, and the shed word stays where it is
/// written.
#[test]
fn one_overload_signal_off_leaves_the_block_on_and_both_off_switch_it_off() {
    let thresholds = |waiting: Option<usize>, usage: Option<f64>| {
        let mut config = config(at_flag_defaults("round_robin"));
        config.worker_overload_waiting_requests = waiting;
        config.worker_overload_token_usage = usage;
        config
    };
    for (waiting, usage) in [
        (None, Some(0.8)),
        (None, Some(0.5)),
        (Some(8), None),
        (Some(32), None),
    ] {
        assert_eq!(
            thresholds(waiting, usage).policy_document(),
            Ok(preset("round_robin")),
            "waiting {waiting:?}, usage {usage:?}"
        );
    }
    let switched_off = edited(preset("round_robin"), |document| {
        document["bindings"][0]["overload"] = json!({ "enabled": false });
    });
    assert_eq!(
        thresholds(None, None).policy_document(),
        Ok(switched_off.clone())
    );
    // With the protection off the thresholds decide nothing: the block
    // is off whatever they are.
    let mut unprotected = thresholds(None, Some(0.8));
    unprotected.worker_overload_protection = false;
    assert_eq!(unprotected.policy_document(), Ok(switched_off));
    // The shed word does not switch the block on.
    let mut shedding = thresholds(None, None);
    shedding.worker_overload_shed = true;
    assert_eq!(
        shedding.policy_document(),
        Ok(edited(preset("round_robin"), |document| {
            document["bindings"][0]["overload"] =
                json!({ "enabled": false, "shed_when_all_vetoed": true });
        }))
    );
}

/// No document carries an overload threshold: one off its default writes
/// nothing, alone and beside the shed word. The switch and the shed word
/// land on the binding, each only off its default.
#[test]
fn an_overload_threshold_lands_on_no_binding_and_the_switch_and_the_shed_word_do() {
    let raised = RoutingSettings {
        worker_overload_waiting_requests: Some(32),
        worker_overload_token_usage: Some(0.9),
        ..config(PolicySettings::RoundRobin)
    };
    assert_eq!(raised.policy_document(), Ok(preset("round_robin")));
    let shedding = RoutingSettings {
        worker_overload_shed: true,
        ..raised
    };
    assert_eq!(
        shedding.policy_document(),
        Ok(edited(preset("round_robin"), |document| {
            document["bindings"][0]["overload"] = json!({ "shed_when_all_vetoed": true });
        })),
    );
    let off = RoutingSettings {
        worker_overload_protection: false,
        ..config(PolicySettings::RoundRobin)
    };
    assert_eq!(
        off.policy_document(),
        Ok(edited(preset("round_robin"), |document| {
            document["bindings"][0]["overload"] = json!({ "enabled": false });
        })),
    );
}

/// A threshold is the gateway's own beside a supplied document too: one
/// off its default says no other document than the settings without it,
/// so the supplied document is in force wherever it is without that
/// threshold.
#[test]
fn an_overload_threshold_beside_a_given_document_is_no_disagreement() {
    let given = preset("least_load");
    for settings in [defaults(), config(at_flag_defaults("least_load"))] {
        let raised = RoutingSettings {
            policy_spec: Some(given.clone()),
            worker_overload_waiting_requests: Some(32),
            worker_overload_token_usage: Some(0.9),
            ..settings
        };
        assert_eq!(
            raised.policy_document_in_force(&defaults()),
            Ok(given.clone())
        );
    }
}

/// The two thresholds are no part of the comparison with a supplied
/// document. Beside settings that state nothing else, a document is in
/// force with one threshold at none and with both at none, whether its
/// block is on or off: what a block that is on means over no threshold
/// is its host's to say. The settings' own document still switches the
/// block off for both at none.
#[test]
fn overload_thresholds_at_none_beside_a_given_document_are_no_disagreement() {
    let on = preset("least_load");
    let off = edited(preset("least_load"), |document| {
        document["bindings"][0]["overload"] = json!({ "enabled": false });
    });
    for (waiting, usage) in [(None, Some(0.9)), (Some(32), None), (None, None)] {
        for given in [&on, &off] {
            let beside = RoutingSettings {
                policy_spec: Some(given.clone()),
                worker_overload_waiting_requests: waiting,
                worker_overload_token_usage: usage,
                ..defaults()
            };
            assert_eq!(
                beside.policy_document_in_force(&defaults()),
                Ok(given.clone()),
                "{waiting:?} and {usage:?}"
            );
        }
    }
    let own = RoutingSettings {
        worker_overload_waiting_requests: None,
        worker_overload_token_usage: None,
        ..defaults()
    };
    // A host whose own defaults state no threshold: the settings' two
    // stand beside a supplied document all the same.
    for (waiting, usage) in [(Some(8), None), (None, Some(0.8)), (Some(8), Some(0.8))] {
        let beside = RoutingSettings {
            policy_spec: Some(on.clone()),
            worker_overload_waiting_requests: waiting,
            worker_overload_token_usage: usage,
            ..defaults()
        };
        assert_eq!(
            beside.policy_document_in_force(&own),
            Ok(on.clone()),
            "{waiting:?} and {usage:?} over a host's defaults at none"
        );
    }
    let block = |document: PolicySpec| {
        document
            .bindings
            .first()
            .and_then(|binding| binding.overload.clone())
    };
    let switched_off = own.policy_document_in_force(&defaults());
    assert_eq!(switched_off, own.policy_document());
    assert_eq!(switched_off.map(block), Ok(block(off)));
}
