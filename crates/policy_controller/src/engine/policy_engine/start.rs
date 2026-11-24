//! The engine's start: the one call a program makes to have an engine.
//!
//! A program that decides in its own process keeps its workers' data where
//! it keeps it for itself: the workers in its registry, their load reports
//! with its load monitor, and each model's cache events in an index.
//! [`PolicyEngine::start`] builds the engine of the program's document and
//! takes each of the three, once ([`WorkerData`]). It answers the engine,
//! which stands for the life of the process, and the engine's own task,
//! which the program spawns.
//!
//! The start makes nothing of its own. The engine is built as
//! [`PolicyEngine::new`] builds it, each hand-over goes through the call
//! that takes it ([`PolicyEngine::set_registry`],
//! [`PolicyEngine::set_loads`], [`Affinity::set_indexes`]), and the task is
//! [`PolicyEngine::run`].
//!
//! [`Affinity::set_indexes`]: super::overlap::Affinity::set_indexes

use std::{future::Future, sync::Arc};

use smg_worker::load::state::LoadReceiver;

use super::{membership::Members, overlap::CacheIndexes, PolicyEngine};
use crate::document::spec::AppliedSpec;

/// Where a program keeps its workers' data: what it hands the engine once,
/// at the engine's start ([`PolicyEngine::start`]).
///
/// The engine reads each in place. A program names every field: a load
/// channel or indexes it does not have are `None`, where a reader of its
/// start sees it.
pub struct WorkerData {
    /// The program's worker registry. The engine's task follows its events
    /// and tells the document's stages a worker that joined or was removed,
    /// by the rule [`PolicyEngine::set_registry`] states. The worker
    /// crate's registry is one.
    pub registry: Arc<dyn Members>,
    /// The program's load channel: its load monitor's subscription. An
    /// engine whose document reads no load report and no named signal, and
    /// none of whose stages keeps a report, keeps none. `None` from a
    /// program that runs no load monitor at its start: a decision whose
    /// request gives no snapshot reads no load report until the program
    /// hands a channel ([`PolicyEngine::set_loads`]).
    pub loads: Option<LoadReceiver>,
    /// The program's indexes of cache events, by model, and with them its
    /// token tree, where it keeps one ([`CacheIndexes::token_tree`]). `None`
    /// from a program that keeps neither: every worker reads as publishing
    /// no cache events, and no request is decided inside a tree of the
    /// program's.
    pub cache_indexes: Option<Arc<dyn CacheIndexes>>,
}

impl PolicyEngine {
    /// Starts the engine of `spec`, the program's document, over the
    /// program's own data. A program has its engine by this call, made once,
    /// at its start.
    ///
    /// It answers two things:
    ///
    /// - The engine, for the life of the process. It is never dropped, so a
    ///   request's guard holds it by a plain reference wherever the request
    ///   travels ([`DecisionGuard`]). A process starts one engine, and
    ///   nothing checks that it does.
    /// - The engine's own task ([`Self::run`]). The program spawns it once,
    ///   ahead of its first request, on a runtime of its choice that has its
    ///   timer on: the engine spawns nothing, and on a runtime with no timer
    ///   the task ends at its first poll with a panic. The task runs the
    ///   engine's upkeep off a request's path, keeps the newest load
    ///   snapshot at hand, tells the document's stages a worker that
    ///   joined or was removed, tells the stages that keep load reports
    ///   each new one, and fills the cells of the worker lists the program
    ///   handed ([`Self::set_lists`]): an engine whose task is never
    ///   spawned does none of the five.
    ///
    /// Every hand-over of `data` is made when the call returns: no decision
    /// of the engine runs ahead of one. The task takes them at its first
    /// poll: it reads the registry whole there, so the document's stages
    /// are told the workers that stand, and takes the load channel, where
    /// the engine kept one. A program relies on both once the task has been
    /// polled once, not when this call returns: a request decided ahead of
    /// that poll reaches stages told no worker, and reads no load report
    /// but one it gives itself.
    ///
    /// Of what a program hands here it hands one thing again: its load
    /// channel, where it builds another load monitor ([`Self::set_loads`]),
    /// since a channel ends with its monitor. The registry and the indexes
    /// are handed here alone, and stand.
    ///
    /// The program's worker lists are not handed here yet: a program that
    /// keeps a cell beside each list hands them by [`Self::set_lists`],
    /// beside this call.
    ///
    /// [`DecisionGuard`]: crate::tracking::DecisionGuard
    #[must_use = "the engine's task does nothing until it is spawned"]
    pub fn start(
        spec: AppliedSpec,
        data: WorkerData,
    ) -> (&'static Self, impl Future<Output = ()> + Send + 'static) {
        // Every field by its name: one added later is handed here, or the
        // build stops.
        let WorkerData {
            registry,
            loads,
            cache_indexes,
        } = data;
        let engine = Self::new(spec);
        engine.set_registry(registry);
        if let Some(loads) = loads {
            engine.set_loads(loads);
        }
        if let Some(indexes) = cache_indexes {
            engine.affinity.set_indexes(indexes);
        }
        let engine: &'static Self = Box::leak(Box::new(engine));
        (engine, engine.run())
    }
}

#[cfg(test)]
mod tests;
