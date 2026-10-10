//! Linux-only Mooncake replay harness. Its measurements rely on CPU affinity,
//! NUMA placement, per-thread resource usage and absolute monotonic sleeps.
#![recursion_limit = "256"]

#[cfg(target_os = "linux")]
mod linux;

#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    linux::main()
}

#[cfg(not(target_os = "linux"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!(
        "mooncake_replay requires Linux for CPU affinity, NUMA policy and resource accounting"
    )
}
