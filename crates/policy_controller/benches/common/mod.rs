//! The synthetic fleet the engine's benches decide over: a per-worker state
//! the way a host reads its routing state, and the view built from it into a
//! reused buffer. Shared by the single-threaded `decision` and `epd` benches
//! and the `contention` bench, so all three measure one fleet.
use smg_policy_controller::{
    Candidate, Engine, Health, LoadRow, PolicySpec, SignalSource, WorkerId,
};

/// What a host reads per worker before it writes a row.
pub struct WorkerState {
    pub id: u64,
    pub healthy: bool,
    pub in_flight: u32,
    pub running: u32,
    pub queued: u32,
    pub kv_permille: u16,
    pub live_rate: u32,
    pub sampled_at_micros: u64,
}

pub fn fleet(n: usize) -> Vec<WorkerState> {
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
    (0..n)
        .map(|i| {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            WorkerState {
                id: 1_000 + i as u64,
                healthy: !x.is_multiple_of(17),
                in_flight: (x >> 8) as u32 % 16,
                running: (x >> 16) as u32 % 12,
                queued: (x >> 24) as u32 % 4,
                kv_permille: ((x >> 32) % 900) as u16,
                live_rate: 1_000 + ((x >> 40) % 3_000) as u32,
                sampled_at_micros: 990_000 + (x >> 48) % 9_000,
            }
        })
        .collect()
}

pub fn build_view(fleet: &[WorkerState], out: &mut Vec<Candidate>) {
    out.clear();
    for (i, w) in fleet.iter().enumerate() {
        out.push(Candidate {
            position: i as u32,
            id: WorkerId(w.id),
            key: w.id as u32,
            health: Health {
                healthy: w.healthy,
                ..Health::default()
            },
            in_flight: w.in_flight,
            load: Some(LoadRow {
                running: w.running,
                queued: w.queued,
                kv_usage_permille: w.kv_permille,
                live_rate: w.live_rate,
                report_age_ms: 5,
                source: SignalSource::Events,
                ..LoadRow::default()
            }),
            load_sampled_at_micros: w.sampled_at_micros,
            ..Candidate::default()
        });
    }
}

/// The engine of one preset; a preset that fails to validate is a bug the
/// unit tests catch.
pub fn engine_of(preset: &str) -> Engine {
    PolicySpec::preset(preset)
        .expect("a preset of this build")
        .validate()
        .expect("a preset validates")
}
