//! The spec store's place on a target that has none.
//!
//! The spec store keeps its generations in a directory, through file calls
//! Linux and macOS give. Every other target compiles this module in its
//! place: the names the door reads, and no value of [`SpecStore`]. A door
//! there has no store to bind, as a host's door in process has none on any
//! target: its writes are refused by name, and no call below ever runs.

use std::{io, time::SystemTime};

use super::Format;

/// One generation on file, as the store's own module names it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Persisted {
    pub(crate) generation: u64,
    pub(crate) format: Format,
}

/// The spec store. It has no value on this target.
pub(crate) enum SpecStore {}

impl SpecStore {
    pub(crate) fn kept(&self) -> &[Persisted] {
        match *self {}
    }

    pub(crate) fn read(&self, _: Persisted) -> io::Result<Vec<u8>> {
        match *self {}
    }

    pub(crate) fn written(&self, _: Persisted) -> io::Result<SystemTime> {
        match *self {}
    }

    pub(crate) fn persist(&mut self, _: Persisted, _: &[u8]) -> io::Result<()> {
        match *self {}
    }

    pub(crate) fn forget(&mut self) -> io::Result<()> {
        match *self {}
    }

    pub(crate) fn prune(&mut self) -> Vec<(u64, io::Result<()>)> {
        match *self {}
    }
}
