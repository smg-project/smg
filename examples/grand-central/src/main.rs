//! A controller of your own: the stock controller with one scorer
//! added in code. The scorer is a type of this file; the built-in stages,
//! the command line, the listeners and the election are the crate's.
//!
//! A change to the view a stage reads ([`Candidate`], [`RequestFacts`]) is a
//! change to what this file compiles against, as it is for a plugin's world.

use std::process::ExitCode;

use smg_grand_central::{
    document::Needs,
    engine::{Candidate, RequestFacts, Scorer, TieBand},
    service::controller,
    stages::{Lifecycle, StageRegistry, StageRegistryError},
};

/// The process's allocator, the one the stock controller's binary registers,
/// on the targets where the settings below hold. An Apple target is left out:
/// the allocator compiles no background thread there. The library registers
/// none: a controller built on it chooses its own.
#[cfg(all(
    not(target_env = "msvc"),
    not(target_env = "musl"),
    not(target_vendor = "apple")
))]
#[global_allocator]
static ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// The allocator's settings, the stock binary's: a background thread hands
/// freed pages back to the system ten seconds after their last use, the
/// threads share sixteen arenas at most, and an allocation up to 32 MiB takes
/// the room the last one freed. The allocator reads them from this symbol as
/// it starts, and the `_RJEM_MALLOC_CONF` environment variable after it,
/// entry by entry.
#[cfg(all(
    not(target_env = "msvc"),
    not(target_env = "musl"),
    not(target_vendor = "apple")
))]
#[expect(
    unsafe_code,
    reason = "the allocator reads its settings from this exported symbol; a NUL-terminated byte string nothing in Rust dereferences"
)]
#[export_name = "_rjem_malloc_conf"]
pub static MALLOC_CONF: &[u8; 100] =
    b"background_thread:true,dirty_decay_ms:10000,muzzy_decay_ms:0,narenas:16,oversize_threshold:33554432\0";

/// Fills a worker before it opens the next: the worker with the most
/// requests running, by its engine's own load report, scores highest, so a
/// fleet under light traffic keeps whole workers idle. A worker with no
/// report yet counts as idle. The profile's filters say where a fill ends:
/// they drop a worker the prober takes out of rotation, and one whose
/// engine reports at or over an overload threshold.
#[derive(Debug)]
struct Pack;

/// A scorer that keeps no state takes every lifecycle hook's default.
impl Lifecycle for Pack {}

impl Scorer for Pack {
    fn name(&self) -> &'static str {
        "pack"
    }

    /// What the host must feed for this scorer to mean anything: the
    /// engines' load reports. A host that feeds a view by its policy's needs
    /// hands none to a scorer that does not declare them, and every worker
    /// then reads as idle.
    fn needs(&self) -> Needs {
        Needs::NONE.with_loads()
    }

    /// This scorer only places, and by the engines' own reports: it turns
    /// no request away. Saying so leaves a price: a stage that says
    /// nothing is taken as one that may refuse a request by what the
    /// gateways report of leases, and each decision under it applies those
    /// reports first.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        // The load row is the engine's own report, as the store holds it.
        let running = |i: u32| {
            candidates[i as usize]
                .load
                .as_ref()
                .map_or(0, |row| row.running)
        };
        let busiest = kept.iter().map(|&i| running(i)).max().unwrap_or(0);
        for (score, &i) in out.iter_mut().zip(kept) {
            // An idle fleet ties: the picker draws among equals.
            *score = if busiest == 0 {
                1.0
            } else {
                running(i) as f32 / busiest as f32
            };
        }
        TieBand::NORMALIZED
    }
}

/// The stock build's stages and this file's scorer, under the name a policy
/// document writes.
fn stages() -> Result<StageRegistry, StageRegistryError> {
    let mut stages = StageRegistry::builtin();
    stages.scorer("pack", || Box::new(Pack))?;
    Ok(stages)
}

/// The crate's command line and run loop over this build's stages.
fn main() -> Result<ExitCode, StageRegistryError> {
    Ok(controller::run(stages()?))
}

#[cfg(test)]
#[path = "../tests/unit/main/tests.rs"]
mod tests;
