use std::{
    pin::Pin,
    task::{Context, Poll},
};

use super::*;

/// Aborting removes the handle, but only dropping the guard finishes a request.
#[test]
fn drain_closes_admission_and_counts_aborted_lifetimes() {
    let registry = Arc::new(RequestRegistry::default());
    let (first, _cancel) = registry.register("first").unwrap();
    assert_eq!(registry.active(), 1);
    registry.abort(&["first".to_string()]).unwrap();
    assert_eq!(registry.len(), 0);
    assert_eq!(registry.active(), 1);
    let (second, _cancel) = registry.register("first").unwrap();
    registry.set_accepting(false);
    assert!(matches!(registry.register("late"), Err(e) if e.code() == tonic::Code::Unavailable));
    drop(first);
    assert_eq!(registry.active(), 1);
    assert_eq!(registry.len(), 1);
    registry.cancel_all().unwrap();
    assert_eq!(registry.active(), 1);
    drop(second);
    assert_eq!(registry.active(), 0);
    registry.set_accepting(true);
    assert!(registry.register("resumed").is_ok());
}

/// The servicer's own probe is abortable like any request and touches
/// the active count neither on registration nor on release: its drop
/// leaves a client's lifetime counted, and the client's drop empties the
/// count.
#[test]
fn an_uncounted_registration_is_abortable_and_outside_the_active_count() {
    let registry = Arc::new(RequestRegistry::default());
    let (client, _cancel) = registry.register("client").unwrap();
    let (probe, mut cancel) = registry.register_uncounted("HEALTH_CHECK_1").unwrap();
    assert_eq!((registry.active(), registry.len()), (1, 2));
    assert_eq!(registry.abort(&["HEALTH_CHECK_1".to_string()]).unwrap(), 1);
    assert!(matches!(cancel.try_recv(), Ok(CancelReason::Abort)));
    drop(probe);
    assert_eq!(
        (registry.active(), registry.len()),
        (1, 1),
        "the probe's release leaves the client's lifetime"
    );
    drop(client);
    assert_eq!(registry.active(), 0);
}

/// While readiness is held for the planned warm-up, the warm-up's own
/// requests are admitted and every other id is refused as not serving;
/// the drain refuses the warm-up's own as well; the release admits all.
#[test]
fn a_readiness_hold_admits_only_the_warm_ups_own_requests() {
    let registry = Arc::new(RequestRegistry::default());
    registry.set_warming(true);
    assert!(registry.is_accepting(), "a hold is not a drain");
    let refused = registry
        .register("chatcmpl-1")
        .err()
        .expect("a foreign request is refused");
    assert_eq!(refused.code(), tonic::Code::Unavailable);
    assert_eq!(refused.message(), "servicer is warming up");
    assert_eq!(registry.active(), 0, "a refused request has no lifetime");
    let (own, _cancel) = registry
        .register(&format!("{WARMUP_REQUEST_ID_PREFIX}7"))
        .expect("the warm-up's own request passes");
    assert_eq!(registry.active(), 1);
    drop(own);
    // The prefix alone is not the form: a caller's id the gateway stamped
    // with its own suffix, or the bare prefix, is a foreign request.
    for crafted in [
        "smg-warmup-x-01a16586-c52f-76e8-bb8f-9f407fd9f836",
        "smg-warmup-1-01a16586-c52f-76e8-bb8f-9f407fd9f836",
        "smg-warmup-",
        "SMG-WARMUP-1",
    ] {
        let refused = registry
            .register(crafted)
            .err()
            .expect("an id that only starts with the prefix is refused");
        assert_eq!(refused.message(), "servicer is warming up", "{crafted}");
    }

    registry.set_accepting(false);
    let drained = registry
        .register(&format!("{WARMUP_REQUEST_ID_PREFIX}8"))
        .err()
        .expect("a drain closes admission to the warm-up as well");
    assert_eq!(drained.message(), "servicer is draining");
    registry.set_accepting(true);

    registry.set_warming(false);
    let (late, _cancel) = registry
        .register("chatcmpl-1")
        .expect("the release admits every request");
    drop(late);
    assert_eq!(registry.active(), 0);
}

/// A registry constructed with the lifecycle owner's nonce admits, while
/// readiness is held, the ids that carry it and nothing else: the bare
/// form a client's own `rid` can spell is refused, as are a wrong nonce,
/// a missing ordinal and a suffix after it; the release admits all.
#[test]
fn a_readiness_hold_admits_only_the_ids_that_carry_the_warm_ups_nonce() {
    let registry = Arc::new(RequestRegistry::with_warmup_nonce(Some(
        "c0ffee42".to_string(),
    )));
    registry.set_warming(true);
    let (own, _cancel) = registry
        .register("smg-warmup-c0ffee42-7")
        .expect("the warm-up's own request carries the nonce");
    drop(own);
    for crafted in [
        "smg-warmup-7",
        "smg-warmup-c0ffee43-7",
        "smg-warmup-c0ffee42-",
        "smg-warmup-c0ffee42-7-8",
        "smg-warmup-c0ffee427",
        "smg-warmup--7",
        "chatcmpl-smg-warmup-c0ffee42-7",
    ] {
        let refused = registry
            .register(crafted)
            .err()
            .expect("an id without the exact nonce form is refused");
        assert_eq!(refused.message(), "servicer is warming up", "{crafted}");
    }
    assert_eq!(registry.active(), 0);
    registry.set_warming(false);
    let (late, _cancel) = registry
        .register("smg-warmup-7")
        .expect("the release admits every request, the bare form too");
    drop(late);
}

/// A choice as the shutdown sees it, in the shape of the servicers' own:
/// a `Complete` queued for the next poll, an engine stream that may hold
/// the engine's parked `Complete`, or no engine stream once the
/// completion is out (or the choice ended). Never polled here.
struct Choice {
    pending: Option<&'static str>,
    inner: Option<Inner>,
}

struct Inner {
    parked: Option<&'static str>,
}

impl Choice {
    fn open() -> Self {
        Self {
            pending: None,
            inner: Some(Inner { parked: None }),
        }
    }

    fn queued(terminal: &'static str) -> Self {
        Self {
            pending: Some(terminal),
            inner: Some(Inner { parked: None }),
        }
    }

    fn parked(terminal: &'static str) -> Self {
        Self {
            pending: None,
            inner: Some(Inner {
                parked: Some(terminal),
            }),
        }
    }

    /// The completion already went out (or the choice ended).
    fn done() -> Self {
        Self {
            pending: None,
            inner: None,
        }
    }
}

impl Stream for Choice {
    type Item = &'static str;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Pending
    }
}

impl HeldTerminal for Choice {
    type Response = &'static str;

    fn finished(&mut self) -> Option<&'static str> {
        if let Some(terminal) = self.pending.take() {
            self.inner = None;
            return Some(terminal);
        }
        let parked = self.inner.as_mut()?.parked.take()?;
        self.inner = None;
        Some(parked)
    }

    fn is_open(&self) -> bool {
        self.pending.is_none() && self.inner.is_some()
    }
}

fn choices(items: Vec<Choice>) -> Choices<Choice> {
    let mut all = Choices::new();
    for item in items {
        all.push(item);
    }
    all
}

#[test]
fn a_held_terminal_goes_out_and_an_open_choice_is_cut() {
    let mut all = choices(vec![Choice::queued("length"), Choice::open()]);
    let (finished, cut) = settle_shutdown(&mut all);
    assert_eq!(finished, ["length"]);
    assert!(cut, "the choice still open is what the shutdown cuts");
    assert_eq!(all.iter().count(), 0, "the choices are dropped afterwards");
}

#[test]
fn a_request_whose_every_choice_had_finished_is_not_cut() {
    let mut all = choices(vec![Choice::queued("stop"), Choice::parked("length")]);
    let (finished, cut) = settle_shutdown(&mut all);
    // `SelectAll` has no order of its own; both completions are there.
    let mut finished: Vec<_> = finished.into_iter().collect();
    finished.sort_unstable();
    assert_eq!(finished, ["length", "stop"]);
    assert!(
        !cut,
        "nothing was open: the stream ends with its completions, no status"
    );
}

#[test]
fn an_ended_choice_neither_yields_nor_cuts() {
    let mut all = choices(vec![Choice::done()]);
    let (finished, cut) = settle_shutdown(&mut all);
    assert!(finished.is_empty());
    assert!(!cut);
}

/// A choice whose completion already went out next to one still
/// generating: only the latter is cut. (That a choice stream drops its
/// engine stream as the engine's own `Complete` goes out, so it is
/// `done` here and not `open`, is pinned per servicer by
/// `a_shutdown_after_the_engines_completion_ends_the_stream_cleanly`.)
#[test]
fn a_choice_whose_completion_already_went_out_is_not_cut() {
    let mut mixed = choices(vec![Choice::done(), Choice::open()]);
    let (finished, cut) = settle_shutdown(&mut mixed);
    assert!(finished.is_empty());
    assert!(cut, "the choice still generating is the one cut");
}
