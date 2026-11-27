//! A document's digest ([`Digest`]) and the canonical form it hashes.

use std::{collections::BTreeMap, fmt, io};

use serde::Serialize;
use sha2::{Digest as _, Sha256};

use super::{
    is_default, BindingSpec, ExplainSpec, IndexSpec, PlansSpec, PolicySpec, ProfileSpec,
    SignalSpec, SpecError, ViewSpec,
};

impl PolicySpec {
    /// The document's digest ([`Digest`]): a change of the file's form or of
    /// the `generation` line leaves it as it was, and any other change to a
    /// document the validator admits moves it. A `kind` line is in no
    /// canonical form and so in no digest: the validator refuses it, and
    /// equal digests say nothing of a document that was not validated.
    pub fn digest(&self) -> Result<Digest, SpecError> {
        let mut hash = Sha256::new();
        serde_json::to_writer(HashWriter(&mut hash), &self.lineless())
            .map_err(|error| SpecError::NoCanonicalForm(error.to_string()))?;
        Ok(Digest(hash.finalize().into()))
    }

    /// The document's canonical form, borrowed: a digest copies none of the
    /// document. Every field is named here, so a field the type gains does
    /// not compile until this form takes it or leaves it out by name.
    fn lineless(&self) -> Lineless<'_> {
        let Self {
            api_version,
            generation: _,
            kind: _,
            view,
            index,
            signals,
            profiles,
            plans,
            bindings,
            explain,
        } = self;
        Lineless {
            api_version,
            generation: 0,
            view,
            index,
            signals,
            profiles,
            plans,
            bindings,
            explain,
        }
    }
}

/// A document's canonical form ([`Digest`]) by reference: [`PolicySpec`]'s
/// fields in its order and under its attributes, the `generation` line
/// zeroed and no `kind` line. It serializes as the document itself does
/// under a zeroed line, byte for byte.
#[derive(Serialize)]
struct Lineless<'a> {
    #[serde(rename = "apiVersion")]
    api_version: &'a str,
    generation: u64,
    #[serde(skip_serializing_if = "is_default::<ViewSpec>")]
    view: &'a ViewSpec,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    index: &'a BTreeMap<String, IndexSpec>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    signals: &'a BTreeMap<String, SignalSpec>,
    profiles: &'a BTreeMap<String, ProfileSpec>,
    #[serde(skip_serializing_if = "is_default::<PlansSpec>")]
    plans: &'a PlansSpec,
    bindings: &'a [BindingSpec],
    #[serde(skip_serializing_if = "is_default::<ExplainSpec>")]
    explain: &'a ExplainSpec,
}

/// A document's digest: SHA-256 over its canonical bytes, which are the
/// document's JSON as [`PolicySpec`] serializes (its maps ordered, its fields
/// in the type's order) with the `generation` line zeroed. Two files that
/// parse into one document share it whatever their form (YAML or JSON, the
/// order of their keys, their comments), and a document applied again under
/// a higher number keeps it. It is written `sha256:` and 64 hex digits.
///
/// The canonical bytes are this build's: two builds whose schema differs may
/// print two digests for one file.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Digest([u8; 32]);

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("sha256:")?;
        self.0.iter().try_for_each(|byte| write!(f, "{byte:02x}"))
    }
}

impl fmt::Debug for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// What a serializer writes, fed to a hash: the canonical bytes are never
/// held whole.
struct HashWriter<'a>(&'a mut Sha256);

impl io::Write for HashWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/document/spec/digest/tests.rs"]
mod tests;
