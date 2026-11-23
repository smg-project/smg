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
pub(crate) mod tests {
    use tonic::{Code, Request};

    use super::{as_told, principal, Principal, ENTRY, NAME_MAX};
    use crate::{
        service::proto::principal::Method,
        store::{token, GatewayId, Registrant},
    };

    /// The refusal of a call with no principal, word for word.
    pub(crate) const NO_PRINCIPAL: &str = "the call has no principal: this listener takes none as told, or the call states no single valid name in x-smg-principal";

    /// A principal as the layer places it.
    pub(crate) fn told(subject: &str) -> Principal {
        Principal {
            subject: subject.to_owned(),
            method: Method::AsTold,
        }
    }

    /// A call that states `name`.
    pub(crate) fn stating(name: &str) -> Request<()> {
        let mut request = Request::new(());
        request
            .metadata_mut()
            .insert(ENTRY, name.parse().expect("an ASCII value"));
        request
    }

    /// One function says the token: a name a principal states is a name a
    /// provider is made under, and no other is.
    #[test]
    fn a_providers_name_is_a_token_a_principal_states() {
        for name in ["cluster", "pods/a-1", "a,b", "a b", ""] {
            let stated = principal(&as_told(stating(name))).is_ok();
            assert_eq!(stated, token(name), "{name:?}");
            assert_eq!(Registrant::provider(name, "k").is_ok(), stated, "{name:?}");
        }
    }

    #[test]
    fn a_call_no_layer_placed_is_refused_as_unauthenticated() {
        // The name is stated and no layer read it: a listener not named.
        let status = principal(&stating("gateway-a")).expect_err("no layer placed one");
        assert_eq!(status.code(), Code::Unauthenticated);
        assert_eq!(status.message(), NO_PRINCIPAL);
    }

    #[test]
    fn the_layer_places_the_stated_name_as_told() {
        let placed = as_told(stating("gateway-a"));
        assert_eq!(
            principal(&placed).expect("the layer placed one"),
            &told("gateway-a")
        );
        // The method's number on the wire is the contract's.
        assert_eq!(i32::from(Method::AsTold), 4);
        // A name at the bound is taken.
        let longest = "a".repeat(NAME_MAX);
        assert_eq!(
            principal(&as_told(stating(&longest)))
                .expect("a name at the bound")
                .subject,
            longest
        );
    }

    #[test]
    fn a_call_that_states_no_name_or_no_usable_one_passes_with_no_principal() {
        let over = "a".repeat(NAME_MAX + 1);
        // No name, an empty one, one over the bound, one with a space.
        for request in [
            Request::new(()),
            stating(""),
            stating(&over),
            stating("gateway a"),
        ] {
            let passed = as_told(request);
            let status = principal(&passed).expect_err("no name to take");
            assert_eq!(status.code(), Code::Unauthenticated);
        }
        // Bytes outside ASCII are no name.
        let mut request = Request::new(());
        request.metadata_mut().insert(
            ENTRY,
            tonic::metadata::MetadataValue::try_from(&b"gateway-\xc3\xa9"[..])
                .expect("an opaque value"),
        );
        assert!(principal(&as_told(request)).is_err());
    }

    #[test]
    fn a_gateways_id_is_its_subjects_fingerprint_and_never_the_embedded_one() {
        let a = told("gateway-a").gateway();
        let b = told("gateway-b").gateway();
        assert_ne!(a, b);
        assert_ne!(a, GatewayId::LOCAL);
        assert_eq!(a, told("gateway-a").gateway());
    }

    /// Two names are no name: neither the first nor the last is taken, so a
    /// hop that adds its own entry to a caller's never decides by position.
    /// Joined into one value, as the wire lets a hop join them, they are two
    /// names still.
    #[test]
    fn a_call_that_states_two_names_passes_with_no_principal() {
        let mut request = stating("gateway-a");
        request
            .metadata_mut()
            .append(ENTRY, "gateway-b".parse().expect("an ASCII value"));
        assert!(principal(&as_told(request)).is_err());
        // The same name twice is two names.
        let mut request = stating("gateway-a");
        request
            .metadata_mut()
            .append(ENTRY, "gateway-a".parse().expect("an ASCII value"));
        assert!(principal(&as_told(request)).is_err());
        // One value with a comma, wherever the comma stands.
        for joined in ["gateway-a,gateway-b", "gateway-a,", ",gateway-a", ","] {
            assert!(principal(&as_told(stating(joined))).is_err(), "{joined}");
        }
    }
}
