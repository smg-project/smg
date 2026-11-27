//! Who a call is from (design 5.2): the principal is transport metadata,
//! never a field a peer fills in. A listener's layer places a [`Principal`]
//! on the call and [`principal`] is its one reader: a call body acts for
//! the peer that function hands it and for no other.
//!
//! A listener has one [`Layer`]. One exists, [`as_told`], on a listener the
//! operator named
//! ([`Controller::identity_as_told`](super::Controller::identity_as_told)).
//! It takes the name a caller states in the [`ENTRY`] metadata entry and
//! verifies nothing: whoever reaches such a listener acts as any principal
//! of its scopes. A listener that was not named has no layer, so every call
//! of its scopes that needs a principal is refused: nothing unplaced is
//! served by default. Where a verifying layer stands, the stated name is
//! never read.

use tonic::{Request, Status};

use crate::{
    engine::view,
    service::proto::principal::Method,
    store::{token, GatewayId},
};

/// The metadata entry a caller states its name in, on a listener that takes
/// identities as told.
pub const ENTRY: &str = "x-smg-principal";

/// The longest name a caller states: a DNS name's bound.
const NAME_MAX: usize = 253;

/// How a listener's calls come by a principal: its one layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Layer {
    /// No layer: no call of the listener has a principal, so each that
    /// needs one is refused.
    Off,
    /// [`as_told`]: the name the caller states, nothing verified.
    AsTold,
}

/// What a listener's layer placed on a call.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Principal {
    /// The subject: on the gateway scope, the gateway's id.
    pub(crate) subject: String,
    /// How the listener came by it: a verification, or as told.
    pub(crate) method: Method,
}

impl Principal {
    /// The subject as the store keys a gateway's leases, vetoes and reports:
    /// its fingerprint, never `0`, which is the embedded mode's own gateway.
    pub(crate) fn gateway(&self) -> GatewayId {
        gateway_id(&self.subject)
    }

    /// The replica itself, for a write of its own discovery: its subject
    /// is the controller's id, and no listener placed it.
    #[cfg(feature = "discovery")]
    pub(crate) fn replica(controller_id: &str) -> Self {
        Self {
            subject: controller_id.to_owned(),
            method: Method::Unspecified,
        }
    }
}

/// The store's key for a gateway's subject.
pub(super) fn gateway_id(subject: &str) -> GatewayId {
    GatewayId(view::fingerprint(subject.as_bytes()))
}

/// The layer of a listener that takes identities as told: the name the call
/// states in [`ENTRY`] becomes its principal, nothing verified. A name is
/// one [`token`] (visible ASCII, 0x21 to 0x7E, with no comma) of at most
/// [`NAME_MAX`] bytes, stated once: a call that states none, two, or one
/// that is no such token passes with no principal. The wire lets a hop
/// join a key's entries into one value with a comma between them, so a
/// comma is two names.
pub(crate) fn as_told(mut request: Request<()>) -> Request<()> {
    let mut stated = request.metadata().get_all(ENTRY).iter();
    let subject = match (stated.next(), stated.next()) {
        (Some(name), None) => name
            .to_str()
            .ok()
            .filter(|name| name.len() <= NAME_MAX && token(name))
            .map(str::to_owned),
        _ => None,
    };
    if let Some(subject) = subject {
        request.extensions_mut().insert(Principal {
            subject,
            method: Method::AsTold,
        });
    }
    request
}

/// The principal a layer placed on `request`, or the contract's refusal of
/// a call with none.
pub(crate) fn principal<T>(request: &Request<T>) -> Result<&Principal, Status> {
    request.extensions().get::<Principal>().ok_or_else(|| {
        Status::unauthenticated(format!(
            "the call has no principal: this listener takes none as told, or the call states no single valid name in {ENTRY}"
        ))
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/service/controller/principal/tests.rs"]
pub(crate) mod tests;
