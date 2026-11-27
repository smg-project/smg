use tikv_jemalloc_ctl::{opt, raw};

#[test]
fn rust_allocations_use_global_jemalloc() {
    let allocated =
        tikv_jemalloc_ctl::thread::allocatedp::read().expect("thread allocation counter");
    let before = allocated.get();
    let allocation = std::hint::black_box(vec![0_u8; 1024 * 1024]);
    let after = allocated.get();
    assert!(after > before, "Rust allocation bypassed jemalloc");
    std::hint::black_box(&allocation);
}

/// The options jemalloc runs with, read back through its control
/// interface: the background purging thread, the dirty and muzzy decay
/// times in milliseconds, the arena count and the size from which an
/// allocation bypasses the arenas' dirty cache. jemalloc skips a
/// misspelled entry with a warning on stderr and runs on, so each one is
/// read back.
fn options_in_effect() -> Result<(bool, isize, isize, u32, usize), tikv_jemalloc_ctl::Error> {
    let background_thread = opt::background_thread::read()?;
    let narenas = opt::narenas::read()?;
    // SAFETY: NUL-terminated option names read into the types jemalloc
    // documents for them (`ssize_t` for the decay times, `size_t` for the
    // threshold).
    let (dirty_decay_ms, muzzy_decay_ms, oversize_threshold) = unsafe {
        (
            raw::read::<isize>(b"opt.dirty_decay_ms\0")?,
            raw::read::<isize>(b"opt.muzzy_decay_ms\0")?,
            raw::read::<usize>(b"opt.oversize_threshold\0")?,
        )
    };
    Ok((
        background_thread,
        dirty_decay_ms,
        muzzy_decay_ms,
        narenas,
        oversize_threshold,
    ))
}

#[test]
fn the_library_applies_the_server_malloc_conf() {
    // The crate's `_rjem_malloc_conf` symbol: purge on a background
    // thread, dirty pages back to the OS after 10 s, muzzy pages at once,
    // 16 arenas instead of four per CPU, and allocations below 32 MiB
    // served from the arenas' dirty cache.
    #[cfg_attr(target_vendor = "apple", expect(unused_variables))]
    let (background_thread, dirty_decay_ms, muzzy_decay_ms, narenas, oversize_threshold) =
        options_in_effect().expect("the options in effect");
    // jemalloc does not compile background-thread support for Mach-O.
    #[cfg(not(target_vendor = "apple"))]
    assert!(
        background_thread,
        "jemalloc did not take the library's malloc_conf: no background thread"
    );
    assert_eq!(
        (dirty_decay_ms, muzzy_decay_ms, narenas, oversize_threshold),
        (10_000, 0, 16, 32 << 20),
        "jemalloc did not take the library's malloc_conf (dirty_decay_ms, muzzy_decay_ms, narenas, oversize_threshold)"
    );
}

#[test]
fn environment_overrides_the_server_malloc_conf_entry_by_entry() {
    // jemalloc reads `_RJEM_MALLOC_CONF` after the exported options, entry
    // by entry; a child process of this test binary runs the probe below
    // under an override of two of them.
    let output = std::process::Command::new(std::env::current_exe().expect("this test binary"))
        .args([
            "--exact",
            "allocator_tests::print_the_options_in_effect",
            "--include-ignored",
            "--nocapture",
        ])
        .env(
            "_RJEM_MALLOC_CONF",
            "background_thread:false,dirty_decay_ms:2000",
        )
        .output()
        .expect("run the probe");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "the probe failed: {stdout}{stderr}"
    );
    assert!(
        stdout.contains(
            "options in effect: background_thread=false dirty_decay_ms=2000 muzzy_decay_ms=0 narenas=16 oversize_threshold=33554432"
        ),
        "_RJEM_MALLOC_CONF did not override the library's malloc_conf entry by entry: {stdout}"
    );
}

/// Run by `environment_overrides_the_server_malloc_conf_entry_by_entry`
/// in a child process; ignored so that a normal run prints nothing.
#[test]
#[ignore = "a probe that environment_overrides_the_server_malloc_conf_entry_by_entry runs in a child process"]
#[expect(
    clippy::print_stdout,
    reason = "the parent test reads this line from the child's stdout"
)]
fn print_the_options_in_effect() {
    let (background_thread, dirty_decay_ms, muzzy_decay_ms, narenas, oversize_threshold) =
        options_in_effect().expect("the options in effect");
    println!(
        "options in effect: background_thread={background_thread} dirty_decay_ms={dirty_decay_ms} muzzy_decay_ms={muzzy_decay_ms} narenas={narenas} oversize_threshold={oversize_threshold}"
    );
}
