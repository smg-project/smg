//! A plugin bound to a profile: [`PluginBinding`], the decision it keeps
//! open on a thread for each phase, and the stage calls that a profile's
//! filter, scorer and picker make through it. [`Binder`] is one engine
//! build's hold on the plugin, and hands out its bindings.
//!
//! This file is a child of the plugin's module on purpose. It reads the
//! plugin's own private items, fields and methods, which a sibling file
//! could read only if each were made visible to it. Do not widen them to
//! move this file out.

use std::{
    cell::{RefCell, RefMut},
    fmt,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};

use thread_local::ThreadLocal;
use wasmtime::component::ResourceAny;

use super::{
    super::{
        bindings::wit,
        config::{EventSet, Failure},
        view::{self as marshal, TooLong},
    },
    bump, thread_cpu_micros, Call, Exports, Fault, Frame, Instance, LoadError, OwnTime, Plugin,
    DESTRUCTOR_BOUND_MICROS, DESTRUCTOR_FRAME_MICROS,
};
use crate::{
    document::spec::Needs,
    engine::{
        decision::{Decision, Target, MAX_TARGETS},
        framework::{self, DecisionScope, TieBand},
        view::{Candidate, Phase, PoolFacts, RequestFacts},
    },
    plugins::shared::{BoundPlugin, Fallback, HalfOpen, OpenDecision},
    tracking::lease::{Lease, Membership, Outcome, ReleaseReason, Side, Signal},
    Draw,
};

/// The decision a bound plugin has open on this thread for one phase: the
/// resource, the budget of its calls, and the answers already taken (a
/// filter answers once per decision, a scorer once; the pair plan repeats
/// the pick per prefill target). The buffers keep their capacity across
/// decisions.
struct Open {
    resource: Option<ResourceAny>,
    /// The generation of the instance that made `resource`.
    generation: u64,
    is_open: bool,
    /// The decision's number, as this open of it returned it.
    decision: u64,
    /// Every stage of this decision takes its fallback: the breaker is open,
    /// the budget is gone, the instance failed, or the request does not pack.
    bypass: bool,
    count: u32,
    /// The fuel cap of one call: the plugin's frame, clamped to the request's
    /// budget (the decision's remainder is the thread's, shared).
    frame_fuel: u64,
    frame_micros: u32,
    /// The decision's clock and draw sequence: what `now-micros` and
    /// `random-u64` read inside this decision's calls, and no other's.
    now: u64,
    draw: Draw,
    /// A declined filter or scorer declines for the whole decision: the
    /// guest is not asked again row by row.
    filter_declined: bool,
    score_declined: bool,
    /// The filter's answer as bits over the candidates, once `mask_set`.
    mask: Vec<u64>,
    mask_set: bool,
    /// The scorer's answer, one per candidate, clamped, once `scores_set`.
    scores: Vec<f32>,
    /// The band the scorer answered beside them, none where it sent none.
    band: Option<f32>,
    scores_set: bool,
    /// The picker's input: the combined scores, `-1.0` on a filtered row.
    scores_in: Vec<f32>,
}

impl Default for Open {
    fn default() -> Self {
        Self {
            resource: None,
            is_open: false,
            decision: 0,
            bypass: false,
            count: 0,
            generation: 0,
            frame_fuel: 0,
            frame_micros: 0,
            now: 0,
            draw: Draw::new(0),
            filter_declined: false,
            score_declined: false,
            mask: Vec::new(),
            mask_set: false,
            scores: Vec::new(),
            band: None,
            scores_set: false,
            scores_in: Vec::new(),
        }
    }
}

impl Open {
    /// Forgets the decision; the buffers keep their capacity.
    fn reset(&mut self) {
        self.resource = None;
        self.is_open = false;
        self.decision = 0;
        self.bypass = false;
        self.count = 0;
        self.generation = 0;
        self.frame_fuel = 0;
        self.frame_micros = 0;
        self.now = 0;
        self.filter_declined = false;
        self.score_declined = false;
        self.mask.clear();
        self.mask_set = false;
        self.scores.clear();
        self.band = None;
        self.scores_set = false;
    }

    /// What this decision's next call runs under, drawing on the thread's
    /// decision budget at the plugin's calibration.
    fn frame<'a>(&'a mut self, budget: &'a mut OpenDecision, fuel_per_micro: u64) -> Frame<'a> {
        Frame {
            cap: self.frame_fuel,
            remaining_micros: &mut budget.remaining_micros,
            fuel_per_micro,
            frame_micros: self.frame_micros,
            now: Some(self.now),
            draw: Some(&mut self.draw),
            quiet: false,
            decision: self.decision,
            seed_micros: budget.seed_micros,
        }
    }

    /// The decision's resource, live on the kept instance: none once the
    /// decision is bypassed, or once the store that made it is gone (a fault
    /// dropped it, and a rebuilt instance would refuse the handle).
    fn live(&self, slot: Option<&Instance>) -> Result<ResourceAny, Fallback> {
        if self.bypass || slot.is_none_or(|i| i.generation != self.generation) {
            return Err(Fallback);
        }
        self.resource.ok_or(Fallback)
    }

    /// An answer outside the contract is a fault of the plugin: the instance
    /// goes, and the rest of this decision falls back.
    fn invalid(&mut self, plugin: &Plugin, instance: &mut Option<Instance>) -> Fallback {
        plugin.fault(instance, Fault::Invalid);
        self.bypass = true;
        self.resource = None;
        Fallback
    }

    /// Reads a call's outcome into the open decision: a failure bypasses the
    /// rest of the decision (the resource went with the instance).
    fn answered<T>(&mut self, call: Call<T>) -> Result<T, Fallback> {
        match call {
            Call::Answered(value) => Ok(value),
            Call::Declined => Err(Fallback),
            Call::BudgetGone => {
                self.bypass = true;
                Err(Fallback)
            }
            Call::Failed => {
                self.bypass = true;
                self.resource = None;
                Err(Fallback)
            }
        }
    }
}

/// One open decision per phase, in the framework's phase slots: the pair
/// plan opens the prefill side and the decode side of one decision, and a
/// binding may name one profile for both, so a bound set keeps the two apart
/// by the phase the stages run under.
#[derive(Default)]
struct Opens([Open; 4]);

/// One engine build's hold on a plugin. The engine hands every lifecycle
/// event to the scorers and the picker of every profile, and a plugin bound
/// to several profiles hears it once: the bound sets a build takes from its
/// one binder share the build's observer, the first scorer or picker adapter
/// made from any of them. A later build over the same loaded plugin (a spec
/// applied again) takes a binder of its own, so its adapters forward for it
/// while the earlier build stands and once it is gone.
#[derive(Debug)]
pub struct Binder {
    pub(super) plugin: Arc<Plugin>,
    pub(super) observer_taken: Arc<AtomicBool>,
}

impl Binder {
    /// Binds the plugin to one profile of this build: the stage set that
    /// shares one `decision` resource per decision.
    pub fn bind(&self) -> Arc<PluginBinding> {
        Arc::new(PluginBinding {
            plugin: Arc::clone(&self.plugin),
            opens: ThreadLocal::new(),
            observer_taken: Arc::clone(&self.observer_taken),
        })
    }
}

/// A plugin bound to one profile: the three stage adapters of a profile
/// share it, so one decision constructs one `decision` resource per phase
/// whichever stages the profile asks of the plugin.
pub struct PluginBinding {
    plugin: Arc<Plugin>,
    opens: ThreadLocal<RefCell<Opens>>,
    /// The build's observer, shared with the build's other bound sets.
    observer_taken: Arc<AtomicBool>,
}

impl fmt::Debug for PluginBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PluginBinding")
            .field("plugin", &self.plugin.name)
            .finish_non_exhaustive()
    }
}

impl PluginBinding {
    pub fn plugin(&self) -> &Arc<Plugin> {
        &self.plugin
    }

    fn state(&self, phase: Phase) -> RefMut<'_, Open> {
        RefMut::map(self.opens.get_or_default().borrow_mut(), |opens| {
            &mut opens.0[framework::slot(phase)]
        })
    }

    /// Opens the decision of the request's phase: the resource over the
    /// view, once per decision (the later stages of the profile find it
    /// open). A gone budget or an open breaker opens nothing: the stages
    /// fall back.
    pub(crate) fn open(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        pool: &PoolFacts<'_>,
    ) {
        let mut open = self.state(facts.phase);
        if open.is_open {
            return;
        }
        open.reset();
        open.is_open = true;
        open.count = u32::try_from(candidates.len()).unwrap_or(u32::MAX);
        let plugin = &self.plugin;
        let frame = plugin.frame;
        open.frame_fuel = frame.fuel_for(facts.budget_micros);
        open.frame_micros = frame.frame_micros(facts.budget_micros);
        open.now = facts.now_micros;
        open.draw = Draw::new(facts.seed);
        // The decision's budget is one a thread: its first open seeds it,
        // every open of the decision (each side, each plugin of every
        // runtime built over the same open decisions) draws on it.
        let mut budget = plugin.runtime.decisions.on_this_thread();
        let decision = budget.open(facts.budget_micros);
        open.decision = decision;
        if plugin.note_open(decision) {
            // The plugin already failed this decision (the other side of a
            // pair, another binding): its part in the decision is over, with
            // no call, no `init` and no second count.
            open.bypass = true;
            return;
        }
        let now = plugin.runtime.now_micros();
        if plugin.breaker.is_open(now) {
            bump(&plugin.tally.bypassed);
            open.bypass = true;
            return;
        }
        if open.frame_fuel == 0 {
            // The frame budget is gone on arrival: no frame is forwarded, no
            // instance is made or fed, the breaker counts a budget hit.
            bump(&plugin.tally.budget_gone);
            plugin.fail(now);
            open.bypass = true;
            return;
        }
        if budget.remaining_micros == 0 {
            // An earlier open of this decision spent the remainder: no frame
            // is forwarded, the stage falls back, the breaker is untouched.
            bump(&plugin.tally.budget_spent);
            open.bypass = true;
            return;
        }
        // A named decider's kept state comes to this thread for the
        // decision; the plugin's last close of it gives it back.
        let working = plugin.working();
        plugin.lend(working, decision);
        let mut guard = working.slot.borrow_mut();
        if guard.is_none() {
            if plugin.emptied.get_or_default().get() {
                // An earlier open or call of this decision dropped the
                // instance: the plugin's part in the decision is over, and
                // the close rebuilds the slot under the full init frame.
                open.bypass = true;
                return;
            }
            // The apply's own instance, where it still waits for this
            // decider: no `init` runs and nothing is charged.
            if let Some(instance) = plugin.adopt() {
                plugin.seat(&mut guard, instance);
            }
        }
        if guard.is_none() {
            // A cold slot: `init` runs under what the decision has left (a
            // slot still empty at the close is rebuilt under the full frame).
            // The budget is wall time here as in a call: the instantiation is
            // charged the larger of its fuel at the calibration and its
            // elapsed time, however it ends. Its own time has two ends like
            // a call's. Nothing watches an `init` (its bound stays on the
            // wall clock), so no tick reads it: its low end is its fuel, and
            // an armed thread reads its clock around it, which makes the two
            // one: a held `init` is no slow sample there.
            let fuel_per_micro = frame.fuel_per_micro.max(1);
            let left = budget.remaining_micros.saturating_mul(fuel_per_micro);
            let init_fuel = plugin.init_fuel();
            let watched = plugin.runtime.thread_clock;
            let start = if watched && plugin.armed() {
                thread_cpu_micros()
            } else {
                None
            };
            let started = Instant::now();
            let (made, used) = plugin.instantiate_metered(init_fuel.min(left));
            let elapsed = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX);
            let fuel_micros = used.div_ceil(fuel_per_micro);
            let charged = fuel_micros.max(elapsed);
            let own = match start.and_then(|start| Some(thread_cpu_micros()?.saturating_sub(start)))
            {
                Some(had) => OwnTime::read(fuel_micros, had, charged),
                None => OwnTime::unread(watched, fuel_micros, charged),
            };
            match made {
                Ok(instance) => {
                    bump(&plugin.tally.lazy_instantiations);
                    budget.remaining_micros = budget.remaining_micros.saturating_sub(charged);
                    plugin.note_init(decision, own);
                    plugin.seat(&mut guard, instance);
                }
                Err(LoadError::Trap { .. })
                    if left < init_fuel && charged >= budget.remaining_micros =>
                {
                    // The remainder clamped `init` below its own frame and
                    // the clamp ended it: the charge reached what was left
                    // (out of fuel uses the whole grant, and the deadline
                    // never fires before the remainder has passed). The
                    // decision's budget is spent (the whole remainder is
                    // drawn, so no later open of the decision runs on it
                    // again), the slot stays cold for the close to rebuild
                    // under the full frame, and the breaker, which judges
                    // `init` under its own frame, is untouched. The plugin's
                    // sample takes the time of this `init` as of any other:
                    // a charge that reached the remainder, so one long on the
                    // clock reads as long here too, unless a reading says
                    // its thread did not have that time.
                    let init_micros = left.div_ceil(fuel_per_micro).max(charged);
                    budget.remaining_micros = budget.remaining_micros.saturating_sub(init_micros);
                    plugin.note_init(decision, own);
                    bump(&plugin.tally.budget_spent);
                    plugin.emptied.get_or_default().set(true);
                    open.bypass = true;
                    return;
                }
                Err(_) => {
                    // `init` failed by itself: under its own frame, or short
                    // of a clamp (a charge below the remainder: it would
                    // have ended the same under the full frame). The failure
                    // is the breaker's, the charge the decision's and part
                    // of the plugin's sample.
                    budget.remaining_micros = budget.remaining_micros.saturating_sub(charged);
                    plugin.note_init(decision, own);
                    plugin.fail(now);
                    open.bypass = true;
                    return;
                }
            }
        }
        // What the membership feed told since this instance last heard comes
        // before the decision reads the instance, as told events: outside
        // the decision's sums. Once a decision, at the first of the plugin's
        // opens that reaches the instance: a later one (a pair's other side,
        // another profile) finds a resource of the decision live on the
        // instance, so a change handed between them waits for the decider's
        // next decision or told event. An open the breaker turned away
        // above reached no instance and is not that first: the instance the
        // decision's next open reaches, or makes, still hears.
        // A change that faults the instance ends the plugin's part in the
        // decision, and the close rebuilds the slot.
        if plugin.feed.is_some() && plugin.heard_in.get_or_default().replace(decision) != decision {
            plugin.catch_up(&mut guard);
            if guard.is_none() {
                open.bypass = true;
                return;
            }
        }
        if let Some(inst) = guard.as_mut() {
            match &inst.exports {
                Exports::Record(_) => inst.records.fill(facts, candidates, pool),
                Exports::Packed(_) => {
                    let filled =
                        inst.packed
                            .fill(facts, candidates, pool, &plugin.exposure, plugin.needs);
                    if filled == Err(TooLong) {
                        bump(&plugin.tally.too_long);
                        open.bypass = true;
                        return;
                    }
                }
            }
        }
        let count = open.count;
        let exposure = &plugin.exposure;
        let needs = plugin.needs;
        let frame = open.frame(&mut budget, frame.fuel_per_micro);
        let constructed = plugin.call(&mut guard, frame, |inst| {
            let result = match &inst.exports {
                Exports::Record(b) => {
                    let headers = marshal::headers(facts, exposure);
                    let request =
                        marshal::request(facts, exposure, needs, &inst.records.tried, &headers);
                    let pool = marshal::pool(pool, &inst.records.zones);
                    b.smg_policy_plugin().decision().call_constructor(
                        &mut inst.store,
                        request,
                        &inst.records.candidates,
                        pool,
                    )
                }
                Exports::Packed(b) => b.smg_policy_plugin_packed().decision().call_constructor(
                    &mut inst.store,
                    &inst.packed.request,
                    &inst.packed.candidates,
                    count,
                    &inst.packed.pool,
                ),
            };
            result.map(Ok)
        });
        match constructed {
            Call::Answered(resource) => {
                open.resource = Some(resource);
                open.generation = guard.as_ref().map_or(0, |i| i.generation);
            }
            Call::Declined | Call::Failed | Call::BudgetGone => open.bypass = true,
        }
    }

    /// Closes every open phase of this thread: the resources are dropped
    /// under their frame's fuel (a destructor is the guest's code too), the
    /// answers forgotten. The scope is closed first: every phase stops
    /// reading as open, and its place in the thread's decision is given
    /// back, before the first destructor runs. A destructor is called under
    /// the host's own code (the clock a build handed the runtime, an
    /// import's body): whatever such a call does, the thread's next decision
    /// opens as its own.
    pub(crate) fn close(&self) {
        /// The destructors' walk. If it does not return, the slot it marked
        /// for the close's rebuild is forgotten: the thread's next decision
        /// finds a cold slot, not one its own calls emptied.
        struct Walk<'a> {
            plugin: &'a Plugin,
            returned: bool,
        }
        impl Drop for Walk<'_> {
            fn drop(&mut self) {
                if !self.returned {
                    self.plugin.emptied.get_or_default().set(false);
                }
            }
        }
        let plugin = &self.plugin;
        let mut opens = self.opens.get_or_default().borrow_mut();
        let mut budget = plugin.runtime.decisions.on_this_thread();
        // One bit a phase that was open.
        let mut closing = 0u8;
        for (phase, open) in opens.0.iter_mut().enumerate() {
            if open.is_open {
                open.is_open = false;
                budget.close();
                closing |= 1 << phase;
            }
        }
        let mut walk = Walk {
            plugin: self.plugin.as_ref(),
            returned: false,
        };
        // The decision these opens are of: each open of it returned the one
        // number.
        let mut decision = 0;
        // Whether the plugin's last open of that decision closes here.
        let mut last = false;
        // The phases that were open, the lowest first.
        while closing != 0 {
            let open = &mut opens.0[closing.trailing_zeros() as usize];
            closing &= closing - 1;
            if let Some(resource) = open.resource.take() {
                let mut guard = plugin.slot().borrow_mut();
                // A resource whose store went with a fault (another binding's
                // close of this decision may have rebuilt the slot since) has
                // nothing to drop: a rebuilt instance would refuse the handle.
                let same_store = guard
                    .as_ref()
                    .is_some_and(|i| i.generation == open.generation);
                if same_store {
                    // The destructor is the guest's code, run for the host's
                    // cleanup: a small fuel frame of its own, outside the
                    // decision's remainder, so a spent budget drops no healthy
                    // instance; a bound on the clock of its own in milliseconds,
                    // so a destructor long on the clock cannot stretch the
                    // decision by a frame; not a latency sample. Running out
                    // of either drops the instance through the call's own
                    // fault path.
                    let mut remaining = u64::from(DESTRUCTOR_BOUND_MICROS);
                    let frame = Frame {
                        cap: plugin.frame.fuel_of(DESTRUCTOR_FRAME_MICROS),
                        remaining_micros: &mut remaining,
                        fuel_per_micro: plugin.frame.fuel_per_micro,
                        frame_micros: DESTRUCTOR_BOUND_MICROS,
                        now: Some(open.now),
                        draw: Some(&mut open.draw),
                        quiet: true,
                        decision: 0,
                        seed_micros: 0,
                    };
                    let _ = plugin.call(&mut guard, frame, |inst| {
                        resource.resource_drop(&mut inst.store).map(Ok)
                    });
                }
            }
            decision = open.decision;
            open.reset();
            last = plugin.note_close(decision, budget.seed_micros);
        }
        walk.returned = true;
        drop(budget);
        // A slot this decision emptied (a call interrupted or trapped, an
        // answer outside the contract, a destructor that ran out) is rebuilt
        // once here, under the full init frame, so the next decision finds it
        // warm: never a slot the decision did not touch (a cold thread's next
        // open instantiates lazily), never while the breaker has the plugin
        // bypassed. Another open of this decision still holding a resource of
        // the dropped store (one plugin, two binding profiles) finds the
        // generation changed.
        if plugin.emptied.get_or_default().replace(false) {
            let now = plugin.runtime.now_micros();
            if !plugin.breaker.is_open(now) {
                let mut guard = plugin.slot().borrow_mut();
                if guard.is_none() {
                    match plugin.instantiate(plugin.init_fuel()) {
                        Ok(instance) => {
                            bump(&plugin.tally.lazy_instantiations);
                            plugin.seat(&mut guard, instance);
                        }
                        Err(error) => {
                            tracing::warn!(plugin = plugin.name, %error, "the rebuild of a dropped instance failed");
                            // One failure a decision: a rebuild that fails after
                            // the fault that emptied the slot was counted is not
                            // a second one.
                            if !plugin.failed_in(decision) {
                                plugin.report(now);
                            }
                        }
                    }
                }
            }
        }
        // The decision is over for the plugin: a named decider's kept state
        // goes back to its rest, rebuilt here where the decision emptied it.
        if last {
            plugin.rest_here();
        }
    }

    /// One guest call of the open decision on this thread's instance, under
    /// the decision's fuel frame: `call` gets the instance and the decision's
    /// live resource. The slot's guard comes back with the answer, for the
    /// caller's own reading of an invalid one; a decision whose resource is
    /// not live falls back with no call made.
    fn ask<T>(
        &self,
        open: &mut Open,
        call: impl FnOnce(&mut Instance, ResourceAny) -> wasmtime::Result<Result<T, wit::Decline>>,
    ) -> Result<(Call<T>, RefMut<'_, Option<Instance>>), Fallback> {
        let plugin = &self.plugin;
        let mut guard = plugin.slot().borrow_mut();
        let resource = open.live(guard.as_ref())?;
        let mut budget = plugin.runtime.decisions.on_this_thread();
        let frame = open.frame(&mut budget, plugin.frame.fuel_per_micro);
        let answer = plugin.call(&mut guard, frame, |inst| call(inst, resource));
        Ok((answer, guard))
    }

    fn ensure_mask(&self, open: &mut Open) -> Result<(), Fallback> {
        if open.mask_set {
            return Ok(());
        }
        if open.filter_declined {
            return Err(Fallback);
        }
        let count = open.count;
        let (answer, mut guard) = self.ask(open, |inst, resource| match &inst.exports {
            Exports::Record(b) => b
                .smg_policy_plugin()
                .decision()
                .call_filter(&mut inst.store, resource),
            Exports::Packed(b) => b
                .smg_policy_plugin_packed()
                .decision()
                .call_filter(&mut inst.store, resource),
        })?;
        open.filter_declined = matches!(answer, Call::Declined);
        let positions = open.answered(answer)?;
        // Unique and in range, else the answer is invalid.
        open.mask.clear();
        open.mask.resize((count as usize).div_ceil(64), 0);
        for &p in &positions {
            if p >= count || bit(&open.mask, p) {
                return Err(open.invalid(&self.plugin, &mut guard));
            }
            open.mask[(p / 64) as usize] |= 1 << (p % 64);
        }
        open.mask_set = true;
        Ok(())
    }
}

impl BoundPlugin for PluginBinding {
    fn name(&self) -> &'static str {
        self.plugin.name()
    }

    fn failure(&self) -> Failure {
        self.plugin.failure()
    }

    fn needs(&self) -> Needs {
        self.plugin.needs()
    }

    fn observes(&self) -> EventSet {
        self.plugin.observes()
    }

    /// The first scorer or picker adapter made from any bound set of the
    /// build forwards the lifecycle; the others do not, so an event reaches
    /// the plugin once per build, whichever of its profiles bind the plugin.
    fn take_observer(&self) -> bool {
        !self.observer_taken.swap(true, Ordering::Relaxed)
    }

    /// The filter's answer for the phase's decision, computed once: `kept`
    /// keeps the positions the guest named.
    fn retain(&self, facts: &RequestFacts<'_>, kept: &mut Vec<u32>) -> Result<(), Fallback> {
        let mut open = self.state(facts.phase);
        self.ensure_mask(&mut open)?;
        kept.retain(|&i| bit(&open.mask, i));
        Ok(())
    }

    /// One row of the filter's answer.
    fn keep(&self, facts: &RequestFacts<'_>, position: u32) -> Result<bool, Fallback> {
        let mut open = self.state(facts.phase);
        self.ensure_mask(&mut open)?;
        Ok(bit(&open.mask, position))
    }

    /// The scorer's answer for the phase's decision, computed once: `out[i]`
    /// is the score of `kept[i]`, and the band is the one the guest answered
    /// beside its scores, the default one where it answered none.
    fn score(
        &self,
        facts: &RequestFacts<'_>,
        kept: &[u32],
        out: &mut [f32],
    ) -> Result<TieBand, Fallback> {
        let mut open = self.state(facts.phase);
        if !open.scores_set {
            if open.score_declined {
                return Err(Fallback);
            }
            let count = open.count;
            let (answer, mut guard) =
                self.ask(&mut open, |inst, resource| match &inst.exports {
                    Exports::Record(b) => b
                        .smg_policy_plugin()
                        .decision()
                        .call_score(&mut inst.store, resource),
                    Exports::Packed(b) => b
                        .smg_policy_plugin_packed()
                        .decision()
                        .call_score(&mut inst.store, resource),
                })?;
            open.score_declined = matches!(answer, Call::Declined);
            let wit::ScoreAnswer {
                mut scores,
                tie_band,
            } = open.answered(answer)?;
            // One finite value per candidate, clamped by the host.
            if scores.len() != count as usize || scores.iter().any(|s| !s.is_finite()) {
                return Err(open.invalid(&self.plugin, &mut guard));
            }
            // A band the guest sends is a finite value from 0 to 1.
            if let Some(band) = tie_band {
                if !(0.0..=1.0).contains(&band) {
                    return Err(open.invalid(&self.plugin, &mut guard));
                }
                open.band = Some(band);
            }
            for s in &mut scores {
                *s = s.clamp(0.0, 1.0);
            }
            // The guest's list is the buffer: no second allocation.
            open.scores = scores;
            open.scores_set = true;
        }
        for (o, &i) in out.iter_mut().zip(kept) {
            *o = open.scores.get(i as usize).copied().unwrap_or(0.0);
        }
        Ok(open.band.map_or(TieBand::NORMALIZED, TieBand))
    }

    /// The picker's answer over the survivors: `scores[i]` belongs to
    /// `kept[i]`; the guest sees one score per candidate, `-1.0` on a row
    /// the filters dropped, and names a surviving position or declines (a
    /// `none` is the pick's decline).
    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        kept: &[u32],
        scores: &[f32],
    ) -> Result<u32, Fallback> {
        let mut open = self.state(facts.phase);
        let count = open.count;
        // A bypassed decision falls back before the buffer is built.
        if open.bypass {
            return Err(Fallback);
        }
        let mut scores_in = std::mem::take(&mut open.scores_in);
        scores_in.clear();
        scores_in.resize(count as usize, -1.0);
        for (&i, &s) in kept.iter().zip(scores) {
            if let Some(slot) = scores_in.get_mut(i as usize) {
                *slot = s;
            }
        }
        let asked = self.ask(&mut open, |inst, resource| {
            let picked = match &inst.exports {
                Exports::Record(b) => b.smg_policy_plugin().decision().call_pick(
                    &mut inst.store,
                    resource,
                    &scores_in,
                ),
                Exports::Packed(b) => b.smg_policy_plugin_packed().decision().call_pick(
                    &mut inst.store,
                    resource,
                    &scores_in,
                ),
            };
            picked.map(|answer| match answer {
                Ok(Some(position)) => Ok(position),
                Ok(None) => Err(wit::Decline::NotApplicable),
                Err(decline) => Err(decline),
            })
        });
        let (answer, mut guard) = match asked {
            Ok(asked) => asked,
            Err(fallback) => {
                // The buffer stays with the decision when no call is made.
                open.scores_in = scores_in;
                return Err(fallback);
            }
        };
        // Within the surviving set, else the answer is invalid.
        let within = match &answer {
            Call::Answered(p) => scores_in.get(*p as usize).is_some_and(|s| *s >= 0.0),
            _ => true,
        };
        open.scores_in = scores_in;
        let picked = open.answered(answer)?;
        if !within {
            return Err(open.invalid(&self.plugin, &mut guard));
        }
        Ok(picked)
    }

    fn observe_issued(&self, lease: &Lease, decision: &Decision) {
        let mut targets: [wit::Target; MAX_TARGETS] = std::array::from_fn(|_| wit::Target {
            id: 0,
            dp_rank: 0,
            pair: None,
        });
        let n = marshal::targets(decision, &mut targets);
        self.plugin.observe(wit::LifecycleEvent::Issued((
            marshal::lease(lease, &self.plugin.exposure),
            &targets[..n],
        )));
    }

    fn observe_dispatched(
        &self,
        lease: &Lease,
        target: &Target,
        side: Side,
        since_decision_micros: u32,
    ) {
        self.plugin
            .observe(wit::LifecycleEvent::Dispatched(wit::Dispatched {
                lease: marshal::lease(lease, &self.plugin.exposure),
                target: marshal::target(target),
                side: marshal::side(side),
                since_decision_micros,
            }));
    }

    fn observe_completed(&self, lease: &Lease, outcome: &Outcome) {
        self.plugin
            .observe(wit::LifecycleEvent::Completed(marshal::completed(
                lease,
                outcome,
                &self.plugin.exposure,
            )));
    }

    fn observe_released(&self, lease: &Lease, reason: ReleaseReason) {
        self.plugin.observe(wit::LifecycleEvent::Released((
            marshal::lease(lease, &self.plugin.exposure),
            marshal::release_reason(reason),
        )));
    }

    fn observe_membership(&self, change: &Membership<'_>) {
        self.plugin.hand(change);
    }

    fn observe_signal(&self, signal: &Signal<'_>) {
        let now = self.plugin.runtime.now_micros();
        self.plugin
            .observe(wit::LifecycleEvent::Signal(marshal::signal(signal, now)));
    }
}

/// This open marks its phase open and counts the decision before its
/// instance's steps, so it runs under the guard that undoes an open that
/// stops half-way.
impl DecisionScope for PluginBinding {
    fn open(&self, facts: &RequestFacts<'_>, candidates: &[Candidate], pool: &PoolFacts<'_>) {
        let mut half_open = HalfOpen::new(self);
        PluginBinding::open(self, facts, candidates, pool);
        half_open.returned();
    }

    fn close(&self) {
        PluginBinding::close(self);
    }
}

fn bit(mask: &[u64], position: u32) -> bool {
    mask.get((position / 64) as usize)
        .is_some_and(|word| word & (1 << (position % 64)) != 0)
}

#[cfg(test)]
mod tests {
    use super::bit;

    #[test]
    fn a_mask_reads_its_bits_and_nothing_past_its_end() {
        let mask = [1u64 << 3, 1];
        assert!(bit(&mask, 3));
        assert!(bit(&mask, 64));
        assert!(!bit(&mask, 4));
        assert!(!bit(&mask, 200));
    }
}
