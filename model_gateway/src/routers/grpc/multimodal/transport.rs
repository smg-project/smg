//! Multimodal tensor transport resolution.
//!
//! Decides how large multimodal tensors travel from the gateway to the worker:
//! the SHM-vs-inline transport mode, its size threshold, the encoder-input wire
//! dtype, and the `/dev/shm` namespace verification that makes the SHM path safe.
//!
//! Resolution precedence for the transport mode and SHM threshold:
//! per-worker `WorkerSpec` override → router config (seeded once at startup via
//! [`init_mm_transport_defaults`]) → `SMG_MM_*` env (with the legacy
//! `SMG_TOKENSPEED_MM_*` names as a fallback) → built-in default (`inline`,
//! 64 KiB).

use std::{
    sync::{Arc, OnceLock},
    time::Duration,
};

use llm_multimodal::Modality;
use openai_protocol::worker::TransportMode;
use smg_mm_rdma::{RdmaConfig, RdmaExporter};
use tracing::{error, info, warn};

use super::settings::mm_settings;
use crate::routers::grpc::{context::WorkerSelection, proto_wrapper::mm_shm_dev_writable};

const DEFAULT_SHM_MIN_BYTES: usize = 64 * 1024;

/// Router-level transport defaults, resolved once at startup from `RouterConfig`
/// (falling back to env, then built-in defaults). Per-worker `WorkerSpec`
/// overrides take precedence over these at request time.
#[derive(Debug, Clone, Copy)]
struct MmTransportDefaults {
    mode: TransportMode,
    shm_min_bytes: usize,
}

static DEFAULTS: OnceLock<MmTransportDefaults> = OnceLock::new();

/// Seed the process-wide transport defaults from router config. Config values
/// win; unset values fall back to env, then the built-in defaults. Call once at
/// startup before serving; idempotent (first call wins).
pub(crate) fn init_mm_transport_defaults(
    mode: Option<TransportMode>,
    shm_min_bytes: Option<usize>,
) {
    let resolved = MmTransportDefaults {
        mode: mode
            .or_else(mm_tensor_transport_mode_from_env)
            .unwrap_or_default(),
        shm_min_bytes: shm_min_bytes
            .or_else(mm_shm_min_bytes_from_env)
            .unwrap_or(DEFAULT_SHM_MIN_BYTES),
    };
    let _ = DEFAULTS.set(resolved);
    log_transport_config_once(resolved);
}

/// The resolved router-level defaults. If [`init_mm_transport_defaults`] was
/// never called (e.g. in tests), resolve lazily from env + built-in defaults.
fn mm_transport_defaults() -> MmTransportDefaults {
    if let Some(defaults) = DEFAULTS.get() {
        return *defaults;
    }
    MmTransportDefaults {
        mode: mm_tensor_transport_mode_from_env().unwrap_or_default(),
        shm_min_bytes: mm_shm_min_bytes_from_env().unwrap_or(DEFAULT_SHM_MIN_BYTES),
    }
}

fn mm_tensor_transport_mode_from_env() -> Option<TransportMode> {
    static LEGACY_WARNED: OnceLock<()> = OnceLock::new();
    let raw = env_with_deprecated_alias(
        "SMG_MM_TENSOR_TRANSPORT",
        "SMG_TOKENSPEED_MM_TENSOR_TRANSPORT",
        &LEGACY_WARNED,
    )?;
    match TransportMode::parse(&raw) {
        Some(mode) => Some(mode),
        None => {
            log_unknown_transport_once(&raw);
            None
        }
    }
}

fn mm_shm_min_bytes_from_env() -> Option<usize> {
    static LEGACY_WARNED: OnceLock<()> = OnceLock::new();
    let raw = env_with_deprecated_alias(
        "SMG_MM_SHM_MIN_BYTES",
        "SMG_TOKENSPEED_MM_SHM_MIN_BYTES",
        &LEGACY_WARNED,
    )?;
    match raw.parse::<usize>() {
        Ok(value) => Some(value),
        Err(_) => {
            log_invalid_shm_min_bytes_once(&raw);
            None
        }
    }
}

/// Read the canonical env var, falling back to the deprecated alias. When the
/// value comes from the alias, log a one-time migration warning (guarded by
/// `warned`, one warning per variable).
fn env_with_deprecated_alias(
    canonical: &str,
    deprecated: &str,
    warned: &OnceLock<()>,
) -> Option<String> {
    if let Some(value) = read_env_nonempty(canonical) {
        return Some(value);
    }
    let value = read_env_nonempty(deprecated)?;
    warned.get_or_init(|| {
        warn!(
            deprecated,
            canonical,
            "Deprecated multimodal transport env var is set; migrate to the canonical name"
        );
    });
    Some(value)
}

fn read_env_nonempty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// Resolve whether large multimodal tensors should use the SHM transport for
/// this request: per-worker override → router default. `shm` forces SHM whenever
/// SMG can write `/dev/shm` (the operator asserts co-location); `auto` also
/// requires the receiving worker leg to be verified as sharing SMG's `/dev/shm`;
/// `inline` (the default) keeps the gRPC path.
pub(super) fn resolve_mm_shm_enabled(
    workers: Option<&WorkerSelection>,
    skip_pixel_values: bool,
) -> bool {
    let mode =
        worker_transport_mode_override(workers).unwrap_or_else(|| mm_transport_defaults().mode);
    match mode {
        TransportMode::Shm => mm_shm_dev_writable(),
        TransportMode::Auto => {
            worker_shares_dev_shm(workers, skip_pixel_values) && mm_shm_dev_writable()
        }
        // `rdma` routes large tensors through the NIXL pixel lane, not SHM.
        TransportMode::Inline | TransportMode::Rdma => false,
    }
}

/// Resolve the SHM size threshold (bytes) for this request: per-worker override
/// → router default.
pub(super) fn resolve_mm_shm_min_bytes(workers: Option<&WorkerSelection>) -> usize {
    worker_shm_min_bytes_override(workers).unwrap_or_else(|| mm_transport_defaults().shm_min_bytes)
}

// ===================== RDMA pixel lane =====================
//
// The gateway owns all RDMA *policy*: it decides whether the lane is on (a
// first-class `TransportMode::Rdma`, with the legacy `SMG_MM_PIXEL_RDMA` env as a
// backward-compatible fallback), parses the env-derived `RdmaConfig`, and builds
// the single process-wide exporter. The engine-neutral `smg-mm-rdma` crate owns
// only the NIXL mechanics + wire format; it reads no env and no globals. In the
// default (stub) build the exporter is inert, so every export falls back to inline.

/// Fixed agent name the encode worker passes to `fetch_remote_metadata`.
const RDMA_GATEWAY_AGENT_NAME: &str = "smg-gateway-encode";
/// Default arena geometry: 64 slots x 32 MiB = 2 GiB host DRAM. A slot must hold
/// one image's framed pixel buffer; raise `SMG_RDMA_SLOT_BYTES` for larger images.
const DEFAULT_RDMA_POOL_SLOTS: usize = 64;
const DEFAULT_RDMA_SLOT_BYTES: usize = 32 * 1024 * 1024;
/// Upper bound on the pre-registered arena (`pool_slots * slot_bytes`). A plausible
/// env misconfiguration (huge `SMG_RDMA_POOL_SLOTS` x `SMG_RDMA_SLOT_BYTES`) would
/// otherwise flow into a single `vec![0u8; total]` whose allocation failure aborts
/// the process instead of falling back to inline. 8 GiB is well above the 2 GiB
/// default and any realistic pool.
const MAX_RDMA_ARENA_BYTES: usize = 8 * 1024 * 1024 * 1024;
/// Fixed slack added to the derived worker-max-hold when deriving the slot TTL.
/// A const rather than an env knob: it only ever widens the lost-notif leak window
/// (a capacity nit, never correctness -- the crate's per-lease gen framing makes a
/// recycled-under-read slot detectable independent of the TTL), and 30s dwarfs any
/// Encode-RPC delivery jitter. `--rdma-slot-ttl-s` / `SMG_RDMA_SLOT_TTL_S` remains
/// the full-TTL override.
const RDMA_SLOT_TTL_SLACK: Duration = Duration::from_secs(30);

/// Process-wide RDMA pixel exporter, built lazily on first use from env-derived
/// config when the RDMA lane is enabled. `None` when the lane is off or NIXL init
/// fails (callers then stay on the inline path).
pub(crate) fn mm_rdma_exporter() -> Option<&'static RdmaExporter> {
    static EXPORTER: OnceLock<Option<RdmaExporter>> = OnceLock::new();
    EXPORTER
        .get_or_init(|| {
            if !rdma_lane_enabled() {
                return None;
            }
            let cfg = build_rdma_config_from_env();
            if cfg.listen_ip.is_empty() {
                // Without a listener IP the worker can't do the cross-node metadata
                // exchange, so every export would fall back to inline anyway. Skip
                // building the NIXL agent + (2 GiB default) arena for nothing.
                warn!(
                    "EPD RDMA: lane enabled but no listener IP (--rdma-listen-ip / SMG_RDMA_LISTEN_IP); staying on the inline path"
                );
                return None;
            }
            match RdmaExporter::new(cfg) {
                Ok(exporter) => Some(exporter),
                Err(e) => {
                    error!(error = %e, "EPD RDMA: exporter init failed; inline fallback");
                    None
                }
            }
        })
        .as_ref()
}

/// Whether the RDMA pixel lane is active: the first-class `TransportMode::Rdma`
/// (`--multimodal-tensor-transport rdma` / `SMG_MM_TENSOR_TRANSPORT=rdma`), with
/// the legacy `--mm-pixel-rdma` / `SMG_MM_PIXEL_RDMA` switch as a fallback.
fn rdma_lane_enabled() -> bool {
    mm_transport_defaults().mode == TransportMode::Rdma || mm_settings().pixel_rdma.value
}

/// Build the exporter config from the `SMG_RDMA_*` env knobs. All RDMA policy lives
/// here; the crate consumes the resulting [`RdmaConfig`] verbatim.
fn build_rdma_config_from_env() -> RdmaConfig {
    // Bound both slot size and count so the arena stays within MAX_RDMA_ARENA_BYTES:
    // a fat-fingered env pair must not turn into an unbounded startup allocation.
    let slot_bytes =
        rdma_env_positive("SMG_RDMA_SLOT_BYTES", DEFAULT_RDMA_SLOT_BYTES).min(MAX_RDMA_ARENA_BYTES);
    let pool_slots = clamp_pool_slots(
        rdma_env_positive("SMG_RDMA_POOL_SLOTS", DEFAULT_RDMA_POOL_SLOTS),
        slot_bytes,
    );
    RdmaConfig {
        // Empty listener IP => the exporter cannot do the cross-node metadata
        // exchange, so the caller stays on the inline path (checked before we build
        // the exporter in `mm_rdma_exporter`).
        listen_ip: mm_settings()
            .rdma_listen_ip
            .value
            .clone()
            .unwrap_or_default(),
        listen_port: rdma_env_parse("SMG_RDMA_LISTEN_PORT", 18515),
        agent_name: RDMA_GATEWAY_AGENT_NAME.to_string(),
        pool_slots,
        slot_bytes,
        slot_ttl: derive_rdma_slot_ttl(),
    }
}

/// Clamp the slot count so the arena (`pool_slots * slot_bytes`) stays within
/// [`MAX_RDMA_ARENA_BYTES`], bounding the startup allocation. Keeps at least one
/// slot; warns when it has to reduce an oversized request.
fn clamp_pool_slots(pool_slots: usize, slot_bytes: usize) -> usize {
    let max_slots = (MAX_RDMA_ARENA_BYTES / slot_bytes.max(1)).max(1);
    if pool_slots > max_slots {
        warn!(
            requested = pool_slots,
            capped = max_slots,
            slot_bytes,
            max_arena_bytes = MAX_RDMA_ARENA_BYTES,
            "EPD RDMA: requested pixel arena exceeds the cap; reducing slot count"
        );
        return max_slots;
    }
    pool_slots
}

/// The worst-case wall time the encode worker may hold a shipped descriptor before
/// and during its one-sided READ: it waits up to `SMG_RDMA_LANDING_WAIT_S` for a
/// free landing slot, then READs for up to `SMG_RDMA_READ_TIMEOUT_S`. These mirror
/// the encode servicer's own knobs (same env names, same defaults) so the two sides
/// cannot drift. The gateway must not reclaim a slot inside this window.
fn worker_max_hold() -> Duration {
    Duration::from_secs(
        rdma_env_parse::<u64>("SMG_RDMA_LANDING_WAIT_S", 120)
            + rdma_env_parse::<u64>("SMG_RDMA_READ_TIMEOUT_S", 60),
    )
}

/// How long a leased slot may live without a free-notif before the reaper
/// force-reclaims it. MUST exceed [`worker_max_hold`] or the TTL races a still-valid
/// READ: the reaper frees the slot, the next image re-leases the SAME address, and
/// the late READ silently returns the WRONG image's pixels. Derived by default
/// (= `worker_max_hold` + [`RDMA_SLOT_TTL_SLACK`]); `--rdma-slot-ttl-s` /
/// `SMG_RDMA_SLOT_TTL_S` overrides, but an override that does not exceed the hold is
/// rejected (see [`resolve_slot_ttl`]).
fn derive_rdma_slot_ttl() -> Duration {
    resolve_slot_ttl(mm_settings().rdma_slot_ttl_s.value, worker_max_hold())
}

/// Apply the TTL invariant to an optional `--rdma-slot-ttl-s` / `SMG_RDMA_SLOT_TTL_S`
/// override: honor it
/// only if it strictly exceeds `hold` (otherwise the reaper could reclaim a slot the
/// worker is still READing and cross-wire images). A too-small override is ignored
/// with a warning in favor of the derived `hold + RDMA_SLOT_TTL_SLACK`. Pure (takes
/// `hold` as a parameter) so the invariant is unit-tested without touching the env.
fn resolve_slot_ttl(override_secs: Option<u64>, hold: Duration) -> Duration {
    if let Some(secs) = override_secs {
        let ttl = Duration::from_secs(secs);
        if ttl > hold {
            return ttl;
        }
        warn!(
            ttl_s = secs,
            hold_s = hold.as_secs(),
            "--rdma-slot-ttl-s / SMG_RDMA_SLOT_TTL_S must exceed the worker's max hold; ignoring override"
        );
    }
    hold + RDMA_SLOT_TTL_SLACK
}

/// Parse a numeric env knob, falling back to `default` when unset or unparsable.
fn rdma_env_parse<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Parse a positive `usize` env knob, falling back to `default` when unset,
/// unparsable, or zero.
fn rdma_env_positive(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

fn worker_transport_mode_override(workers: Option<&WorkerSelection>) -> Option<TransportMode> {
    primary_worker(workers)?
        .metadata()
        .spec
        .multimodal_tensor_transport
}

fn worker_shm_min_bytes_override(workers: Option<&WorkerSelection>) -> Option<usize> {
    primary_worker(workers)?
        .metadata()
        .spec
        .multimodal_shm_min_bytes
}

/// The worker whose per-worker overrides apply. Multimodal tensors are sent to
/// wherever the vision encoder runs: the encode worker in EPD (so its spec wins),
/// otherwise the single/prefill worker that does the encoding itself.
fn primary_worker(workers: Option<&WorkerSelection>) -> Option<&Arc<dyn crate::worker::Worker>> {
    match workers? {
        WorkerSelection::Single { worker } => Some(worker),
        WorkerSelection::Disaggregated {
            encode_assignments,
            prefill,
            ..
        } => encode_assignments
            .as_ref()
            .and_then(|assignments| assignments.first())
            .map(|assignment| &assignment.worker)
            .or(Some(prefill)),
    }
}

/// The wire dtype for the vLLM encoder input.
///
/// Defaults to float32 rather than to the narrower width the TokenSpeed path
/// prefers: a worker only reads the bytes at the width it already knows, so the
/// router keeps sending the widest one until it is told the other end reads
/// something else.
pub(super) fn mm_vllm_encoder_input_dtype(workers: Option<&WorkerSelection>) -> String {
    resolve_mm_vllm_encoder_input_dtype(
        mm_vllm_encoder_input_dtype_from_env(),
        mm_encoder_input_dtype_from_worker(workers),
    )
}

fn resolve_mm_vllm_encoder_input_dtype(
    override_dtype: Option<String>,
    worker_dtype: Option<String>,
) -> String {
    override_dtype
        .or(worker_dtype)
        .unwrap_or_else(|| "float32".to_string())
}

fn mm_vllm_encoder_input_dtype_from_env() -> Option<String> {
    static DTYPE: OnceLock<Option<String>> = OnceLock::new();
    cached_env_dtype(&DTYPE, "SMG_VLLM_ENCODER_INPUT_DTYPE")
}

pub(super) fn mm_encoder_input_dtype(
    modality: Modality,
    workers: Option<&WorkerSelection>,
) -> String {
    resolve_mm_encoder_input_dtype(
        mm_modality_encoder_input_dtype_from_env(modality),
        mm_default_encoder_input_dtype_from_env(),
        mm_encoder_input_dtype_from_worker(workers),
    )
}

fn resolve_mm_encoder_input_dtype(
    modality_override: Option<String>,
    default_override: Option<String>,
    worker_dtype: Option<String>,
) -> String {
    // Use one configurable wire policy across modalities, with an optional
    // per-modality override for encoder contracts that require another dtype.
    modality_override
        .or(default_override)
        .or(worker_dtype)
        .unwrap_or_else(|| "bfloat16".to_string())
}

fn mm_modality_encoder_input_dtype_from_env(modality: Modality) -> Option<String> {
    static IMAGE_DTYPE: OnceLock<Option<String>> = OnceLock::new();
    static VIDEO_DTYPE: OnceLock<Option<String>> = OnceLock::new();
    static AUDIO_DTYPE: OnceLock<Option<String>> = OnceLock::new();

    match modality {
        Modality::Image | Modality::ImageEmbeds => {
            cached_env_dtype(&IMAGE_DTYPE, "SMG_TOKENSPEED_IMAGE_ENCODER_INPUT_DTYPE")
        }
        Modality::Video => {
            cached_env_dtype(&VIDEO_DTYPE, "SMG_TOKENSPEED_VIDEO_ENCODER_INPUT_DTYPE")
        }
        Modality::Audio => {
            cached_env_dtype(&AUDIO_DTYPE, "SMG_TOKENSPEED_AUDIO_ENCODER_INPUT_DTYPE")
        }
    }
}

fn mm_default_encoder_input_dtype_from_env() -> Option<String> {
    static DEFAULT_DTYPE: OnceLock<Option<String>> = OnceLock::new();
    cached_env_dtype(&DEFAULT_DTYPE, "SMG_TOKENSPEED_ENCODER_INPUT_DTYPE")
}

fn cached_env_dtype(cell: &'static OnceLock<Option<String>>, name: &str) -> Option<String> {
    cell.get_or_init(|| std::env::var(name).ok().filter(|dtype| !dtype.is_empty()))
        .clone()
}

fn mm_encoder_input_dtype_from_worker(workers: Option<&WorkerSelection>) -> Option<String> {
    primary_worker(workers)?
        .metadata()
        .spec
        .labels
        .get("multimodal_encoder_dtype")
        .filter(|dtype| !dtype.is_empty())
        .cloned()
}

fn log_transport_config_once(defaults: MmTransportDefaults) {
    static LOGGED: OnceLock<()> = OnceLock::new();
    LOGGED.get_or_init(|| {
        info!(
            mode = %defaults.mode,
            shm_min_bytes = defaults.shm_min_bytes,
            dev_writable = mm_shm_dev_writable(),
            "Multimodal tensor transport configured"
        );
    });
}

fn log_unknown_transport_once(value: &str) {
    static WARNED: OnceLock<()> = OnceLock::new();
    WARNED.get_or_init(|| {
        warn!(
            value,
            "Unknown multimodal tensor transport value; expected inline|shm|auto|rdma, using inline"
        );
    });
}

fn log_invalid_shm_min_bytes_once(value: &str) {
    static WARNED: OnceLock<()> = OnceLock::new();
    WARNED.get_or_init(|| {
        warn!(
            value,
            "Invalid multimodal SHM min-bytes value; expected a non-negative integer, using default"
        );
    });
}

/// Whether the worker is *verified* to share SMG's `/dev/shm`, making the SHM
/// transport safe for this payload.
///
/// Rather than inferring locality from the worker URL (TCP loopback proves only
/// network locality, not a shared `/dev/shm`), the worker advertises its
/// `/dev/shm` filesystem identity (`<boot_id>:<st_dev of /dev/shm>`) via
/// `GetServerInfo`, which discovery stores in the worker's `shm_namespace_id`
/// label. Two processes share `/dev/shm` iff these tokens match: `boot_id` pins
/// the host, and `st_dev` is the tmpfs superblock device, identical whenever the
/// same tmpfs backs both `/dev/shm` mounts — including separate containers that
/// share it via `--ipc`/bind-mount (where mount-namespace inodes differ but the
/// underlying superblock is the same). We compare the worker's token to ours:
/// equal ⇒ shared. A missing/empty token or any mismatch is treated as
/// non-sharing, so `auto` safely falls back to inline.
fn worker_shares_dev_shm(workers: Option<&WorkerSelection>, skip_pixel_values: bool) -> bool {
    let Some(local) = local_shm_namespace_id() else {
        return false;
    };
    match workers {
        Some(WorkerSelection::Single { worker }) => worker_matches_shm_namespace(worker, local),
        Some(WorkerSelection::Disaggregated {
            encode_assignments,
            prefill,
            decode,
            ..
        }) => {
            if !skip_pixel_values {
                if let Some(encode_assignments) = encode_assignments {
                    // EPD: encoder_input (pixels) ships gateway -> encode worker, so SHM
                    // is safe only if every encode worker assigned in this request shares
                    // the gateway's /dev/shm. A mixed local/remote fan-out must fall back
                    // to inline/RDMA rather than giving a remote worker an unreadable SHM handle.
                    return encode_assignments
                        .iter()
                        .all(|assignment| worker_matches_shm_namespace(&assignment.worker, local));
                }
            }
            worker_matches_shm_namespace(prefill, local)
                && worker_matches_shm_namespace(decode, local)
        }
        None => false,
    }
}

fn worker_matches_shm_namespace(worker: &Arc<dyn crate::worker::Worker>, local: &str) -> bool {
    worker
        .metadata()
        .spec
        .labels
        .get("shm_namespace_id")
        .is_some_and(|id| !id.is_empty() && id == local)
}

/// This process's `/dev/shm` filesystem identity: `<boot_id>:<st_dev of /dev/shm>`.
/// `boot_id` pins the host (it is not namespaced) and `st_dev` is the tmpfs
/// superblock device backing `/dev/shm`; together they identify the tmpfs so two
/// processes sharing it (even across containers via `--ipc`/bind-mount) produce
/// the same token. Computed once; `None` if it can't be determined (then `auto`
/// stays inline).
fn local_shm_namespace_id() -> Option<&'static str> {
    static ID: OnceLock<Option<String>> = OnceLock::new();
    ID.get_or_init(compute_shm_namespace_id).as_deref()
}

#[cfg(unix)]
fn compute_shm_namespace_id() -> Option<String> {
    use std::os::unix::fs::MetadataExt;
    let boot_id = std::fs::read_to_string("/proc/sys/kernel/random/boot_id").ok()?;
    let shm_dev = std::fs::metadata("/dev/shm").ok()?.dev();
    Some(format!("{}:{shm_dev}", boot_id.trim()))
}

#[cfg(not(unix))]
fn compute_shm_namespace_id() -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modality_dtype_precedes_global_then_worker_with_bfloat16_fallback() {
        assert_eq!(
            resolve_mm_encoder_input_dtype(
                Some("float16".to_string()),
                Some("float32".to_string()),
                Some("bfloat16".to_string()),
            ),
            "float16"
        );
        assert_eq!(
            resolve_mm_encoder_input_dtype(
                None,
                Some("float32".to_string()),
                Some("bfloat16".to_string()),
            ),
            "float32"
        );
        assert_eq!(
            resolve_mm_encoder_input_dtype(None, None, Some("bfloat16".to_string()),),
            "bfloat16"
        );
        assert_eq!(resolve_mm_encoder_input_dtype(None, None, None), "bfloat16");
    }

    /// The vLLM wire starts from the widest dtype and narrows only when asked,
    /// the opposite of the TokenSpeed default above. A worker reads the bytes at
    /// whatever width it knows, so guessing a narrower one turns every media
    /// request against an older worker into a failure or worse.
    #[test]
    fn vllm_dtype_stays_float32_until_something_asks_for_narrower() {
        assert_eq!(resolve_mm_vllm_encoder_input_dtype(None, None), "float32");
        assert_eq!(
            resolve_mm_vllm_encoder_input_dtype(None, Some("bfloat16".to_string())),
            "bfloat16"
        );
        assert_eq!(
            resolve_mm_vllm_encoder_input_dtype(
                Some("float16".to_string()),
                Some("bfloat16".to_string()),
            ),
            "float16"
        );
    }

    /// The derived slot TTL must strictly exceed the worker's max hold, so the
    /// reaper can never reclaim a slot the worker could still be reading (a late
    /// READ against a recycled slot cross-wires images). The crate's `SlotPool`
    /// tests cover the mechanics; this pins the gateway's TTL-derivation policy.
    #[test]
    fn derived_rdma_slot_ttl_exceeds_worker_max_hold() {
        assert!(
            derive_rdma_slot_ttl() > worker_max_hold(),
            "slot_ttl {:?} must exceed worker_max_hold {:?} or a late READ cross-wires",
            derive_rdma_slot_ttl(),
            worker_max_hold()
        );
    }

    /// The `SMG_RDMA_SLOT_TTL_S` override is honored only when it exceeds the worker
    /// hold; a too-small (or absent) value falls back to the derived `hold + slack`,
    /// so an operator can never silently reintroduce the recycled-under-READ bug.
    #[test]
    fn slot_ttl_override_must_exceed_hold() {
        let hold = Duration::from_secs(180);
        // Override above the hold is honored verbatim.
        assert_eq!(resolve_slot_ttl(Some(600), hold), Duration::from_secs(600));
        // Override at or below the hold is rejected -> derived hold + slack.
        assert_eq!(
            resolve_slot_ttl(Some(180), hold),
            hold + RDMA_SLOT_TTL_SLACK
        );
        assert_eq!(resolve_slot_ttl(Some(10), hold), hold + RDMA_SLOT_TTL_SLACK);
        // No override -> derived.
        assert_eq!(resolve_slot_ttl(None, hold), hold + RDMA_SLOT_TTL_SLACK);
    }

    /// A slot count that would blow past the arena cap is reduced to fit (>= 1),
    /// while a normal request passes through untouched.
    #[test]
    fn pool_slots_capped_to_arena_max() {
        let slot_bytes = 1024 * 1024 * 1024; // 1 GiB
        let capped = clamp_pool_slots(1_000_000, slot_bytes);
        assert_eq!(capped, MAX_RDMA_ARENA_BYTES / slot_bytes);
        assert!(capped >= 1, "must keep at least one slot");
        assert_eq!(clamp_pool_slots(64, 32 * 1024 * 1024), 64);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn local_shm_namespace_id_resolves_on_linux() {
        // /proc/.../boot_id and /dev/shm both exist on the Linux CI/runtime
        // image, so the token must resolve to `<boot_id>:<st_dev>`. If it ever
        // returned None, `auto` would silently never enable SHM.
        let id = local_shm_namespace_id().expect("shm namespace id should resolve on Linux");
        assert!(
            id.contains(':'),
            "token must be <boot_id>:<st_dev>, got {id:?}"
        );
        let dev = id.rsplit(':').next().unwrap();
        assert!(
            dev.parse::<u64>().is_ok(),
            "st_dev component must be numeric, got {id:?}"
        );
    }
}
