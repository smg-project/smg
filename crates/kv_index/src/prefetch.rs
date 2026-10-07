//! A read-prefetch hint, so a batch of hash-table probes can have its cache misses overlap: call
//! it on the home slots of the keys a few steps ahead, then probe.
//!
//! A prefetch is advice to the cache, never a load: it cannot fault, cannot change program state,
//! and may be ignored by the hardware. That is why [`prefetch_read`] is a safe function for any
//! pointer, aligned or not, mapped or not, dangling or null. The function is a no-op on targets
//! without a prefetch instruction.
//!
//! This is the crate's one unsafe line, behind a safe function; the workspace denies unsafe code
//! and the exception is scoped to that function, not to a crate of its own.

#![deny(unsafe_op_in_unsafe_fn)]

/// Hint that the cache line at `pointer` will be read soon (first-level cache, keep). Sound for
/// any pointer: a prefetch instruction never faults and has no effect on program state.
#[inline(always)]
#[cfg_attr(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    expect(
        unsafe_code,
        reason = "a prefetch instruction is a hint: it never faults, reads nothing, writes nothing"
    )
)]
pub fn prefetch_read<T>(pointer: *const T) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: `prfm pldl1keep` is a hint that never faults, whatever the address holds; it
        // reads no memory, writes no memory and no register, and touches no flags.
        unsafe {
            core::arch::asm!(
                "prfm pldl1keep, [{address}]",
                address = in(reg) pointer,
                options(nostack, preserves_flags, readonly)
            );
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `prefetcht0` is a hint that never faults, whatever the address holds.
        unsafe {
            use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            _mm_prefetch::<{ _MM_HINT_T0 }>(pointer.cast::<i8>());
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = pointer;
    }
}

#[cfg(test)]
mod tests {
    use super::prefetch_read;

    #[test]
    fn prefetching_a_stack_value_is_harmless() {
        let value = [7u64; 16];
        prefetch_read(value.as_ptr());
        prefetch_read(&value[15]);
        assert_eq!(value[3], 7);
    }

    #[test]
    fn prefetching_dangling_and_null_pointers_does_not_fault() {
        let dangling = 0x7f00_0000_0000usize as *const u64;
        prefetch_read(dangling);
        prefetch_read(core::ptr::null::<u64>());
        prefetch_read(usize::MAX as *const u8);
    }
}
