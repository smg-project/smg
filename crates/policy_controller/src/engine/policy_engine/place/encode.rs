//! An encode request placed: a decision over its caller's three lists, each
//! item's encoder, and the first of its pair's targets both of whose
//! workers can be sent to. Where an encoder or every target cannot be sent
//! to, the call decides again as a pair's does ([`place`]).
//!
//! The engine's entry for an encode request decides the pair and each
//! item's encoder in one call ([`Loaded::decide_encode_at`]). The
//! targets of its pick share one prefill, and an item's encoder hands to
//! that prefill. So the pair's targets are walked as a pair's are
//! ([`sides`]), and the encoders stand beside whichever is taken.
//!
//! An item has one encoder and no second. Each is read once, by the one
//! read a single request's worker gets ([`ready`]), inside the walk: what
//! is read of a host's worker under a lease stands under the walk's catch.
//! The route's own rule is
//! asked of the pair's two workers and of no encoder. Where an encoder
//! cannot be sent to, no target is taken: the decision is given up by
//! name, and the encoder joins the workers the call's next decision leaves
//! out, with each side of the pair that cannot be sent to either.
//!
//! Three lists with no item stated are placed as their pair
//! ([`super::pair`]), and name no encoder.

use std::{
    cell::{Cell, OnceCell, RefCell},
    sync::Arc,
    time::Instant,
};

use super::{
    pair::{self, decode_label, sides},
    ready, walk, Call, Made, PlaceRequest, Unplaced,
};
use crate::{
    engine::{
        framework::{EncodeItems, EncodePlaced},
        policy_engine::Loaded,
        prefills_full, EncodeLists,
    },
    tracking::{Encoders, Placement},
};

thread_local! {
    /// Where a thread's encode decisions put each item's encoder. Its room
    /// is kept: a placement allocates for the encoders it names alone.
    pub(super) static PLACED: RefCell<EncodePlaced> = RefCell::new(EncodePlaced::new());
}

/// One ask of `call` for an encode request over `lists`, the three lists
/// its caller handed, with the `items` it carries: one decision told to
/// leave out the workers the call missed so far ([`Call::missed`]), each
/// item's encoder, and the first of the pair's targets both of whose
/// workers can be sent to.
///
/// The placement names the request's routing value, the profile of its
/// decode side by its label, and each item's encoder.
///
/// `None` where an item's encoder cannot be sent to, or no target of the
/// pair holds. The decision is then given up by name, and each worker that
/// could not be sent to joins the call's missed workers by its own id: an
/// encoder's, a prefill's or a decode's. The pair's sides are read behind
/// an encoder that cannot be sent to as well, so one ask notes them all.
///
/// # Errors
///
/// The decision's shed, its queue or its refusal. Prefills with an open
/// partner that are all at the caller's admission limit answer
/// [`Unplaced::PrefillsFull`], as a pair's do.
pub(super) fn ask<'a>(
    engine: Loaded<'static, '_>,
    request: &PlaceRequest<'a>,
    lists: &EncodeLists<'_>,
    items: EncodeItems<'_>,
    call: &mut Call,
) -> Result<Option<Placement<'a>>, Unplaced> {
    PLACED.with(|placed| match placed.try_borrow_mut() {
        Ok(mut placed) => asked(engine, request, lists, items, call, &mut placed),
        // A stage placed a request from inside a decision of this thread:
        // that request's encoders take a list of their own.
        Err(_) => asked(
            engine,
            request,
            lists,
            items,
            call,
            &mut EncodePlaced::new(),
        ),
    })
}

/// [`ask`], with `placed` to take each item's encoder.
fn asked<'a>(
    engine: Loaded<'static, '_>,
    request: &PlaceRequest<'a>,
    lists: &EncodeLists<'_>,
    items: EncodeItems<'_>,
    call: &mut Call,
    placed: &mut EncodePlaced,
) -> Result<Option<Placement<'a>>, Unplaced> {
    let made = Made::asked(engine.object, request, call);
    let dated = Cell::new(None);
    let clock = || {
        let now = Instant::now();
        dated.set(Some(now));
        now
    };
    let decided = engine.decide_encode_at(&made, items, placed, clock);
    let (decided, later) = decided.map_err(Unplaced::Refused)?;
    let (key, kept) = (made.key(), [made.sent.get(), made.decodes.get()]);
    if prefills_full(&decided.decision) {
        return Err(Unplaced::PrefillsFull);
    }
    let now = dated.get().unwrap_or_else(Instant::now);
    // Each item's encoder, and the id of each one that cannot be sent to.
    // A decision that placed nothing names none.
    let encoders = || {
        let mut workers = Vec::with_capacity(placed.encoders().len());
        let mut down = Vec::new();
        for encoder in placed.encoders() {
            let worker = lists.encoders.workers.get(encoder.position as usize);
            match worker.filter(|worker| ready(worker.as_ref())) {
                Some(worker) => workers.push(Arc::clone(worker)),
                None => down.push(encoder.id),
            }
        }
        (workers, down)
    };
    // The encoders are read once, at the walk's first target, so under its
    // catch: a leased decision is given up as a panic passes. Behind an
    // encoder that cannot be sent to no target is taken: the walk reads
    // each target's sides all the same, notes the ones that cannot be sent
    // to, and gives the decision up.
    let read = OnceCell::new();
    let taken = walk(engine, decided, later, now, &mut call.missed, |target| {
        let (_, down) = read.get_or_init(&encoders);
        let sides = sides(request, &lists.pair, target, &kept)?;
        if down.is_empty() {
            Ok(sides)
        } else {
            Err([None; 2])
        }
    })?;
    let (workers, down) = read.into_inner().unwrap_or_default();
    for id in down {
        if !call.missed.contains(&id) {
            call.missed.push(id);
        }
    }
    Ok(taken.map(|mut placement| {
        placement.routing_key = key;
        placement.decode_profile = Some(decode_label(engine, request.model));
        // A decision that placed no item names no encoder, and allocates
        // nothing for one.
        if !workers.is_empty() {
            placement.encoders = Encoders::named(workers, placed.one());
        }
        placement
    }))
}

/// The asks of one call for an encode request over `lists` ([`ask`]): where
/// an ask takes no placement, the call's next ask answers in its place,
/// as a pair's does ([`Call::again`]).
///
/// # Errors
///
/// What an ask answers, and [`Unplaced::NoWorkers`] where the call's last
/// ask takes no placement.
pub(super) fn place<'a>(
    engine: Loaded<'static, '_>,
    request: &PlaceRequest<'a>,
    lists: &EncodeLists<'_>,
    items: EncodeItems<'_>,
    call: &mut Call,
) -> Result<Placement<'a>, Unplaced> {
    match ask(engine, request, lists, items, call)? {
        Some(placement) => Ok(placement),
        None => call.again(|call| place(engine, request, lists, items, call)),
    }
}

/// A request that hands an encode request's three lists: placed with the
/// items it carries ([`place`]), and as its pair where it states none.
/// Out of line: a request with other lists runs none of it.
///
/// # Errors
///
/// What the call for its items or for its pair answers.
#[inline(never)]
pub(super) fn three<'a>(
    engine: Loaded<'static, '_>,
    request: &PlaceRequest<'a>,
    lists: &EncodeLists<'_>,
) -> Result<Placement<'a>, Unplaced> {
    match request.encode {
        Some(items) => place(engine, request, lists, items, &mut Call::default()),
        // No item stated: the three lists are placed as their pair.
        None => pair::place(engine, request, &lists.pair, &mut Call::default()),
    }
}
