//! EPD RDMA pixel transport: the gateway stages each image's serialized pixel
//! buffer into a pre-registered host-DRAM arena and hands the worker a small wire
//! descriptor; the worker PULLs the pixels with a one-sided RDMA READ instead of
//! receiving them inline in the Encode gRPC frame. On the inline path, serializing
//! the large pixel buffer burns gateway CPU and starves the encode worker's GPU --
//! the EPD bottleneck this transport removes.
//!
//! One host-DRAM arena is registered with NIXL ONCE at [`RdmaExporter::new`] and
//! sub-divided into fixed slots. Per image the hot path only LEASES a free slot,
//! frames the pixels as `[gen u64][pixels][gen u64]`, and ships
//! `[magic 8B][slot_addr u64 LE][gen u64 LE][slot_key i64 LE][port u16 LE][ip utf8]`
//! -- NO per-image register_memory and NO growing agent metadata. The worker fetches
//! the gateway's (fixed) metadata ONCE, then READs slot offsets into its own
//! pre-registered landing pool. The transfer notif is tagged with the slot key; the
//! reaper consumes it to return the slot to the free list (a TTL sweep is the
//! lost-notif net).
//!
//! This crate owns only the NIXL mechanics: it reads no environment and no globals.
//! The gateway builds a [`RdmaConfig`] (all policy) and constructs one exporter.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use nixl_sys::{
    Agent, AgentConfig, Backend, MemType, MemoryRegion, NixlDescriptor, NixlError, NotificationMap,
    OptArgs, RegistrationHandle,
};
use parking_lot::Mutex;
use tracing::debug;

use crate::{slot_pool::SlotPool, RdmaConfig, DESCRIPTOR_MAGIC, GEN_BYTES};

/// Reaper poll interval.
const REAPER_TICK: Duration = Duration::from_millis(20);

/// Failure constructing a [`RdmaExporter`].
#[derive(Debug)]
pub enum RdmaError {
    /// NIXL agent init or arena registration failed.
    Nixl(NixlError),
    /// The background reaper thread could not be spawned. Without it, leased slots
    /// are never reclaimed, so the pool would exhaust; the gateway must fall back to
    /// inline rather than run with a dead reaper.
    Reaper(std::io::Error),
}

impl std::fmt::Display for RdmaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RdmaError::Nixl(e) => write!(f, "RDMA exporter init failed: {e:?}"),
            RdmaError::Reaper(e) => write!(f, "RDMA reaper thread spawn failed: {e}"),
        }
    }
}

impl std::error::Error for RdmaError {}

impl From<NixlError> for RdmaError {
    fn from(e: NixlError) -> Self {
        RdmaError::Nixl(e)
    }
}

/// NIXL descriptor over the pre-registered arena (host DRAM). Registered once; the
/// backing allocation is leaked (process-lifetime) so `base` is stable forever.
#[derive(Debug)]
struct ArenaRegion {
    base: usize,
    size: usize,
}

impl MemoryRegion for ArenaRegion {
    #[expect(
        unsafe_code,
        reason = "NIXL MemoryRegion trait method is unsafe; body only reinterprets the stored base address as a pointer"
    )]
    unsafe fn as_ptr(&self) -> *const u8 {
        self.base as *const u8
    }
    fn size(&self) -> usize {
        self.size
    }
}

impl NixlDescriptor for ArenaRegion {
    fn mem_type(&self) -> MemType {
        MemType::Dram
    }
    fn device_id(&self) -> u64 {
        0
    }
}

/// Pre-registered host-DRAM arena: the pure [`SlotPool`] plus the single persistent
/// NIXL registration that keeps the backing memory pinned for the worker's READ.
struct SlotArena {
    pool: SlotPool,
    /// The single persistent registration; kept alive for the arena's life.
    _handle: RegistrationHandle,
}

/// Persistent gateway NIXL agent (one per exporter). The agent is touched only at
/// init (register the arena) and by the reaper (drain notifs); the hot path is
/// agent-free, so it stays behind a coarse Mutex without contending the fast path.
struct GatewayRdma {
    agent: Agent,
    // The UCX backend. We must pass it explicitly to register_memory (OptArgs) so the
    // arena gets a UCX/rc rkey: register_memory(None) leaves the registration not bound
    // to UCX, which is fine for TCP (copy) but makes a one-sided rc READ hang (no rkey).
    backend: Backend,
}

/// Shared state the hot path and the reaper both touch. Held behind an `Arc` so the
/// detached reaper keeps it alive for the process (matching the original leaked
/// static state); the gateway holds the exporter in a process-lifetime singleton.
struct ExporterInner {
    cfg: RdmaConfig,
    agent: Mutex<GatewayRdma>,
    arena: SlotArena,
}

/// Stages preprocessed pixel buffers into a pre-registered NIXL arena for one-sided
/// RDMA READ by the worker. Construct once (per process) via [`RdmaExporter::new`].
pub struct RdmaExporter {
    inner: Arc<ExporterInner>,
}

impl RdmaExporter {
    /// Build the persistent NIXL agent (UCX backend), register the slot arena, and
    /// spawn the reaper. Returns `Err` if NIXL init or the arena registration fails
    /// (the gateway then leaves callers on the inline path).
    pub fn new(cfg: RdmaConfig) -> Result<Self, RdmaError> {
        let gw = init_agent(&cfg)?;
        let agent = Mutex::new(gw);
        let arena = build_arena(&agent, &cfg)?;
        let inner = Arc::new(ExporterInner { cfg, agent, arena });
        spawn_reaper(Arc::clone(&inner)).map_err(RdmaError::Reaper)?;
        debug!(
            slots = inner.cfg.pool_slots,
            slot_bytes = inner.cfg.slot_bytes,
            ttl_s = inner.cfg.slot_ttl.as_secs(),
            "EPD RDMA: gateway NIXL agent + UCX backend up, pixel arena registered"
        );
        Ok(RdmaExporter { inner })
    }

    /// Stage `bytes` (the serialized pixel buffer for one image) into a pre-registered
    /// arena slot keyed by `slot_key`, and return the wire descriptor for the puller:
    /// `[magic 8B][slot_addr u64 LE][gen u64 LE][slot_key i64 LE][port u16 LE][ip utf8]`.
    /// The worker fetches the gateway's (fixed) metadata once, connects to the listener
    /// (ip:port), then READs `nbytes + FRAME_OVERHEAD` from `slot_addr` and re-checks
    /// the stamps against `gen`. The slot returns to the free list on the worker's
    /// free-notif or the TTL. On any failure (no listener IP, oversized image, or no
    /// free slot) returns `Err(bytes)` so the caller re-attaches them inline.
    pub fn export(&self, slot_key: i64, bytes: Vec<u8>) -> Result<Vec<u8>, Vec<u8>> {
        self.export_slice(slot_key, &bytes).ok_or(bytes)
    }

    /// Stage an immutable byte slice in the slot arena.
    pub fn export_slice(&self, slot_key: i64, bytes: &[u8]) -> Option<Vec<u8>> {
        let inner = &self.inner;
        if inner.cfg.listen_ip.is_empty() {
            // No listener IP -> cannot do the cross-node metadata exchange.
            return None;
        }
        // An oversized tensor or an exhausted pool leaves the payload inline.
        let (slot, addr, gen) =
            inner
                .arena
                .pool
                .lease_and_write(slot_key, bytes, Instant::now())?;

        let ip = inner.cfg.listen_ip.as_bytes();
        let port = inner.cfg.listen_port;
        let mut descriptor =
            Vec::with_capacity(DESCRIPTOR_MAGIC.len() + 8 + GEN_BYTES + 8 + 2 + ip.len());
        descriptor.extend_from_slice(DESCRIPTOR_MAGIC);
        descriptor.extend_from_slice(&addr.to_le_bytes());
        descriptor.extend_from_slice(&gen.to_le_bytes());
        descriptor.extend_from_slice(&slot_key.to_le_bytes());
        descriptor.extend_from_slice(&port.to_le_bytes());
        descriptor.extend_from_slice(ip);
        debug!(
            slot_key,
            addr, slot, "EPD RDMA: staged pixel slot (listener {}:{})", inner.cfg.listen_ip, port
        );
        Some(descriptor)
    }
}

fn init_agent(cfg: &RdmaConfig) -> Result<GatewayRdma, NixlError> {
    // enable_listen_thread + a fixed listen_port so the encode worker (initiator)
    // can do the bidirectional NIXL metadata exchange against this gateway (target).
    // Without the listener the worker's connect hits the ephemeral worker port ->
    // "Connection refused" cross-node. enable_prog_thread (default) keeps the UCX
    // worker progressing so one-sided READs complete without our intervention.
    let agent_cfg = AgentConfig {
        enable_listen_thread: true,
        listen_port: i32::from(cfg.listen_port),
        ..Default::default()
    };
    let agent = Agent::new_configured(&cfg.agent_name, &agent_cfg)?;
    let (_mems, params) = agent.get_plugin_params("UCX")?;
    let backend = agent.create_backend("UCX", &params)?;
    Ok(GatewayRdma { agent, backend })
}

fn build_arena(agent: &Mutex<GatewayRdma>, cfg: &RdmaConfig) -> Result<SlotArena, NixlError> {
    let total = cfg.pool_slots.saturating_mul(cfg.slot_bytes);
    // Leak the arena for the process lifetime: `base` must stay registered + stable,
    // and the gateway agent never shuts down. Holding it as raw memory (not a typed
    // Box) is what makes the per-slot memcpy through `base` sound.
    let boxed = vec![0u8; total].into_boxed_slice();
    let base = Box::into_raw(boxed) as *mut u8 as usize;
    let region = ArenaRegion { base, size: total };
    let register = || -> Result<RegistrationHandle, NixlError> {
        let guard = agent.lock();
        // Bind the registration to the UCX backend so it gets an rc rkey (needed for
        // the worker's one-sided RDMA READ; without it the READ hangs over rc).
        let mut opt = OptArgs::new()?;
        opt.add_backend(&guard.backend)?;
        guard.agent.register_memory(&region, Some(&opt))
    };
    match register() {
        Ok(handle) => Ok(SlotArena {
            pool: SlotPool::new(base, cfg.slot_bytes, cfg.pool_slots),
            _handle: handle,
        }),
        Err(e) => {
            // Registration failed: reclaim the arena we just leaked rather than
            // burning up to `pool_slots * slot_bytes` (2 GiB by default) for the
            // process lifetime while the gateway falls back to the inline path.
            // SAFETY: `base`/`total` are exactly the pointer+length produced by the
            // `Box::into_raw` above and nothing else references the buffer yet
            // (registration failed, no slot leased), so reconstructing and dropping
            // the Box is sound.
            #[expect(
                unsafe_code,
                reason = "free the arena leaked via Box::into_raw when NIXL registration fails"
            )]
            unsafe {
                drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                    base as *mut u8,
                    total,
                )));
            }
            Err(e)
        }
    }
}

/// Background reaper: drain free-notifs (tag = slot key) to return the leased slot to
/// the free list, plus a TTL sweep so a lost notif can never leak a slot. Holds an
/// `Arc<ExporterInner>` so it runs for the process lifetime. Returns the spawn error
/// so [`RdmaExporter::new`] can fail (and the gateway stay inline) rather than run
/// with a dead reaper that would leak every leased slot until the pool exhausts.
fn spawn_reaper(inner: Arc<ExporterInner>) -> std::io::Result<()> {
    std::thread::Builder::new()
        .name("epd-rdma-reaper".into())
        .spawn(move || loop {
            std::thread::sleep(REAPER_TICK);
            if let Ok(mut notifs) = NotificationMap::new() {
                {
                    let guard = inner.agent.lock();
                    let _ = guard.agent.get_notifications(&mut notifs, None);
                }
                if let Ok(map) = notifs.take_notifs() {
                    for (_agent, tags) in map {
                        for tag in tags {
                            if let Ok(key) = tag.parse::<i64>() {
                                inner.arena.pool.free_slot_key(key);
                            }
                        }
                    }
                }
            }
            // TTL sweep: reclaim slots whose READ-notif never arrived. `slot_ttl` is
            // derived (by the gateway) to exceed the worker's max hold, so this never
            // races a live READ.
            let _ = inner
                .arena
                .pool
                .reap_stale(Instant::now(), inner.cfg.slot_ttl);
        })?;
    Ok(())
}
