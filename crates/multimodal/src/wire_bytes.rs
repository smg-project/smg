//! Owned wire bytes and immutable views of a cacheable wire allocation.

use std::{
    mem,
    ops::{Deref, Range},
    sync::Arc,
};

use bytes::Bytes;

use crate::buffer_pool;

/// Wire bytes that can be handed to a transport or viewed by a cache entry.
#[derive(Clone, Debug)]
pub struct WireBytes(Storage);

#[derive(Clone, Debug)]
enum Storage {
    Owned(Vec<u8>),
    Shared(Shared),
}

#[derive(Clone, Debug)]
struct Shared {
    allocation: Arc<Allocation>,
    range: Range<usize>,
}

#[derive(Debug)]
struct Allocation(Vec<u8>);

impl Drop for Allocation {
    fn drop(&mut self) {
        buffer_pool::recycle(mem::take(&mut self.0));
    }
}

impl AsRef<[u8]> for Shared {
    fn as_ref(&self) -> &[u8] {
        &self.allocation.0[self.range.clone()]
    }
}

impl From<Vec<u8>> for WireBytes {
    fn from(bytes: Vec<u8>) -> Self {
        Self(Storage::Owned(bytes))
    }
}

impl Default for WireBytes {
    fn default() -> Self {
        Vec::new().into()
    }
}

impl PartialEq for WireBytes {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Deref for WireBytes {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl WireBytes {
    /// View the logical bytes, without unused allocation capacity.
    pub fn as_slice(&self) -> &[u8] {
        match &self.0 {
            Storage::Owned(bytes) => bytes,
            Storage::Shared(bytes) => bytes.as_ref(),
        }
    }

    /// Hand the bytes to an inline transport without copying either form.
    pub fn into_payload(self) -> Bytes {
        match self.0 {
            Storage::Owned(bytes) => buffer_pool::recycled_bytes(bytes),
            Storage::Shared(bytes) => Bytes::from_owner(bytes),
        }
    }

    /// Return an owned buffer immediately, or release the shared view.
    pub fn recycle(self) {
        if let Storage::Owned(bytes) = self.0 {
            buffer_pool::recycle(bytes);
        }
    }

    /// Identity and capacity of the full allocation a shared view retains.
    /// The identity stays valid while at least one view is alive.
    pub(crate) fn shared_allocation(&self) -> Option<(usize, usize)> {
        let Storage::Shared(bytes) = &self.0 else {
            return None;
        };
        Some((
            Arc::as_ptr(&bytes.allocation) as usize,
            bytes.allocation.0.capacity(),
        ))
    }

    pub(crate) fn shared(bytes: Vec<u8>) -> Self {
        let range = 0..bytes.len();
        Self(Storage::Shared(Shared {
            allocation: Arc::new(Allocation(bytes)),
            range,
        }))
    }

    pub(crate) fn shared_slice(&self, range: Range<usize>) -> Option<Self> {
        let Storage::Shared(bytes) = &self.0 else {
            return None;
        };
        if range.start > range.end || range.end > bytes.range.len() {
            return None;
        }
        Some(Self(Storage::Shared(Shared {
            allocation: Arc::clone(&bytes.allocation),
            range: bytes.range.start + range.start..bytes.range.start + range.end,
        })))
    }

    pub(crate) fn join_adjacent(&mut self, other: &Self) -> bool {
        let (Storage::Shared(left), Storage::Shared(right)) = (&mut self.0, &other.0) else {
            return false;
        };
        if !Arc::ptr_eq(&left.allocation, &right.allocation) || left.range.end != right.range.start
        {
            return false;
        }
        left.range.end = right.range.end;
        true
    }
}
