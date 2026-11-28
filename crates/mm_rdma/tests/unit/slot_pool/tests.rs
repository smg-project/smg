use super::*;

/// Build a `SlotPool` over a real (leaked) heap arena -- no NIXL agent, no RDMA.
/// Leaking matches production (the arena is process-lifetime) and keeps the raw
/// `base` valid for the whole test.
fn test_pool(n_slots: usize, slot_bytes: usize) -> SlotPool {
    let buf = vec![0u8; n_slots * slot_bytes].into_boxed_slice();
    let base = Box::into_raw(buf) as *mut u8 as usize;
    SlotPool::new(base, slot_bytes, n_slots)
}

/// Read back what physically sits at a slot address (what a one-sided READ to that
/// address would return).
fn bytes_at(addr: u64, len: usize) -> Vec<u8> {
    #[expect(
        unsafe_code,
        reason = "read bytes back from a NIXL slot address for gen verification"
    )]
    unsafe {
        std::slice::from_raw_parts(addr as *const u8, len).to_vec()
    }
}

/// Reproduces the bug: a too-short TTL reclaims a slot while the worker is still
/// within its hold window, the next image re-leases the SAME address, and a late
/// READ against the old descriptor returns the WRONG image's bytes.
#[test]
fn short_ttl_recycles_slot_under_a_live_descriptor() {
    let pool = test_pool(1, 64);
    let t0 = Instant::now();
    let img_a = b"AAAAAAAAAAAAAAAA";
    let (_slot_a, addr_a, _gen_a) = pool.lease_and_write(1, img_a, t0).expect("lease A");

    // Worker has NOT issued its READ yet (e.g. parked on landing-ring backpressure).
    // The reaper runs with a 30s TTL after 31s -> reclaims the still-referenced slot.
    let freed = pool.reap_stale(t0 + Duration::from_secs(31), Duration::from_secs(30));
    assert_eq!(
        freed, 1,
        "30s TTL reclaims a slot still under a live descriptor"
    );

    // The next image re-leases the freed slot -> same physical address.
    let img_b = b"BBBBBBBBBBBBBBBB";
    let (_slot_b, addr_b, _gen_b) = pool
        .lease_and_write(2, img_b, t0 + Duration::from_secs(31))
        .expect("lease B");
    assert_eq!(addr_a, addr_b, "freed slot's address is reused");

    // The late READ for key 1 reads addr_a, whose payload (at +GEN_BYTES) now holds
    // key 2's pixels. The TTL no longer protects this; the gen stamps do (next test).
    assert_eq!(
        bytes_at(addr_a + GEN_BYTES as u64, img_b.len()),
        img_b,
        "stale READ returns the WRONG image's payload (physical cross-wire remains)"
    );
}

/// The gen guard: a recycled slot carries a fresh gen, so a worker validating the
/// slot's header/trailer stamps against its (older) descriptor gen detects the
/// cross-wire and fails the key -- independent of the TTL value.
#[test]
fn gen_guard_detects_recycled_slot() {
    let pool = test_pool(1, 64);
    let t0 = Instant::now();
    let img_a = b"AAAAAAAA";
    let (_a, addr_a, gen_a) = pool.lease_and_write(1, img_a, t0).expect("lease A");

    pool.reap_stale(t0 + Duration::from_secs(31), Duration::from_secs(30));
    let img_b = b"BBBBBBBB";
    let (_b, addr_b, gen_b) = pool
        .lease_and_write(2, img_b, t0 + Duration::from_secs(31))
        .expect("lease B");
    assert_eq!(addr_a, addr_b, "slot address is physically reused");
    assert_ne!(gen_a, gen_b, "each lease gets a fresh, larger generation");

    // The slot's header+trailer stamps now read gen_b. A worker holding key 1's
    // descriptor (gen_a) compares against these and sees a mismatch -> reject.
    let hdr = u64::from_le_bytes(bytes_at(addr_a, GEN_BYTES).try_into().unwrap());
    let trl = u64::from_le_bytes(
        bytes_at(addr_a + (GEN_BYTES + img_b.len()) as u64, GEN_BYTES)
            .try_into()
            .unwrap(),
    );
    assert_eq!(hdr, gen_b);
    assert_eq!(trl, gen_b);
    assert_ne!(
        hdr, gen_a,
        "stamp != descriptor gen -> worker fails the key"
    );
}

/// A normal lease frames the payload as `[gen][payload][gen]` with both stamps
/// equal to the returned (and shipped-in-descriptor) gen.
#[test]
fn gen_framing_brackets_the_payload() {
    let pool = test_pool(1, 64);
    let t0 = Instant::now();
    let (_s, addr, gen) = pool.lease_and_write(7, b"PIXELS!!", t0).expect("lease");
    assert_eq!(
        u64::from_le_bytes(bytes_at(addr, GEN_BYTES).try_into().unwrap()),
        gen,
        "header stamp == gen"
    );
    assert_eq!(bytes_at(addr + GEN_BYTES as u64, 8), b"PIXELS!!", "payload");
    assert_eq!(
        u64::from_le_bytes(
            bytes_at(addr + (GEN_BYTES + 8) as u64, GEN_BYTES)
                .try_into()
                .unwrap()
        ),
        gen,
        "trailer stamp == gen"
    );
}

/// With a TTL that exceeds the worker's max hold, a reap at the hold does NOT
/// reclaim, so the address cannot be reused under a live descriptor: a second image
/// finds the pool full and the gateway falls back to inline, never aliasing the
/// slot. (The gateway derives `ttl > hold`; here we pass explicit stand-ins.)
#[test]
fn derived_ttl_keeps_slot_through_worker_hold() {
    let pool = test_pool(1, 64);
    let t0 = Instant::now();
    let hold = Duration::from_secs(180);
    let ttl = hold + Duration::from_secs(30);
    let (_slot_a, addr_a, _gen_a) = pool
        .lease_and_write(1, b"AAAAAAAAAAAAAAAA", t0)
        .expect("lease A");

    let freed = pool.reap_stale(t0 + hold, ttl);
    assert_eq!(
        freed, 0,
        "ttl > worker hold must not reclaim within the hold"
    );

    // Slot still leased -> no free slot -> caller goes inline, no address reuse.
    assert!(
        pool.lease_and_write(2, b"BBBBBBBBBBBBBBBB", t0 + hold)
            .is_none(),
        "leased slot is not handed to a second image"
    );
    assert_eq!(
        bytes_at(addr_a + GEN_BYTES as u64, 16),
        b"AAAAAAAAAAAAAAAA",
        "the slot still holds key 1's pixels (payload at +GEN_BYTES) for its READ"
    );
}

/// The recycle is not a one-slot artifact: with two slots, both reclaim under a
/// short TTL and a later lease reuses one of the freed addresses.
#[test]
fn multi_slot_addresses_recycle_after_reclaim() {
    let pool = test_pool(2, 64);
    let t0 = Instant::now();
    let (_a, addr_a, _ga) = pool.lease_and_write(1, b"A", t0).unwrap();
    let (_b, addr_b, _gb) = pool.lease_and_write(2, b"B", t0).unwrap();
    assert_ne!(addr_a, addr_b, "distinct leases get distinct addresses");

    let freed = pool.reap_stale(t0 + Duration::from_secs(31), Duration::from_secs(30));
    assert_eq!(freed, 2);

    let (_c, addr_c, _gc) = pool
        .lease_and_write(3, b"C", t0 + Duration::from_secs(31))
        .unwrap();
    assert!(
        addr_c == addr_a || addr_c == addr_b,
        "a reclaimed address is reused by the next lease"
    );
}

/// An oversized image never leases (caller -> inline), and a normal free returns
/// the slot for reuse.
#[test]
fn oversize_rejected_and_free_returns_slot() {
    // Slot holds FRAME_OVERHEAD (16) of gen stamps + payload, so 24 bytes => 8 of
    // usable payload.
    let pool = test_pool(1, FRAME_OVERHEAD + 8);
    let t0 = Instant::now();
    assert!(
        pool.lease_and_write(1, &[0u8; 9], t0).is_none(),
        "payload + gen frame larger than a slot is rejected"
    );
    let (_s, _addr, _g) = pool
        .lease_and_write(1, &[1u8; 8], t0)
        .expect("fits exactly");
    assert!(
        pool.lease_and_write(2, &[2u8; 8], t0).is_none(),
        "pool now full"
    );
    pool.free_slot_key(1);
    assert!(
        pool.lease_and_write(2, &[2u8; 8], t0).is_some(),
        "freed slot is reusable"
    );
}
