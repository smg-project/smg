//! One generation of the policy document as a store applies it
//! ([`AppliedSpec`]): the engine the document validates into, and what a host
//! reads of the document beside the engine. The schema and its validator
//! stand in [`spec`](super::spec), and this module calls that validator's
//! build of a document.

#[cfg(feature = "in-process")]
use std::collections::BTreeMap;
#[cfg(feature = "controller")]
use std::sync::atomic::{self, AtomicBool};
use std::sync::Arc;

// The key source's block is named only by a doc link of this module.
#[cfg(doc)]
use crate::document::spec::KeySourceSpec;
#[cfg(feature = "in-process")]
use crate::document::spec::{index_matching, IndexSpec, ProfileSpec, SignalSpec};
#[cfg(feature = "in-process")]
use crate::engine::sticky_sessions::SessionTable;
use crate::{
    document::{
        routing_key::KeySources,
        spec::{
            DecodePick, Digest, HashDepth, HashDepths, Needs, OverloadSwitch, Pairing, PolicySpec,
            SpecError, ViewLists, ViewSpec, STOCK,
        },
    },
    engine::{
        framework::Engine,
        view::{blocks_within, HashKind, Phase},
    },
    load::signals::SignalColumn,
    policies::cache_aware::Gates,
    stages::registry::StageRegistry,
};

/// What a spec's profiles were built from: each profile's block by its
/// name, and the two blocks every profile's build reads beside its own.
#[cfg(feature = "in-process")]
#[derive(Debug, Default)]
struct BuiltFrom {
    profiles: BTreeMap<String, ProfileSpec>,
    view: ViewSpec,
    signals: BTreeMap<String, SignalSpec>,
}

/// One generation as a store applies it: the engine validated from a
/// document, with the generation's number, its needs and its view's cap on
/// a request's block hashes.
///
/// One invariant: **a decision reads its gateway's vetoes, its engine and
/// its number in one load.** A store publishes every world under one of these
/// ([`StoreSnapshot::spec`](crate::store::StoreSnapshot::spec)), so no reader
/// pairs one generation's engine with rows published under another. The number
/// is the engine's own, the document's `generation` line: there is no second
/// holder.
///
/// A document becomes one by validation alone ([`AppliedSpec::validate`]):
/// one that fails validation never does. The other constructor validates
/// nothing: [`AppliedSpec::unbound`] is what a store holds before any
/// document is applied, and the only generation 0 a store ever holds. A
/// document that names no generation validates to number 0 too, and the
/// store's apply refuses it as not above the standing one.
///
/// It carries the document's digest beside the number ([`Digest`]): the
/// number says which generation a replica runs, the digest which document,
/// and one load reads the pair.
///
/// It also holds what pair classes are derived under, beside the engine: the
/// document's pairing mode ([`AppliedSpec::pairing`]). It says when a host
/// asks a pair's decode ([`AppliedSpec::decode_pick`]). And it says how deep
/// a host hashes a request for it ([`AppliedSpec::block_hashes_read`]).
#[derive(Debug)]
pub struct AppliedSpec {
    engine: Engine,
    /// What the engine's profiles were built from
    /// ([`AppliedSpec::keep_from`]).
    #[cfg(feature = "in-process")]
    built_from: BuiltFrom,
    /// The digest of the document the engine was validated from; none for
    /// the unbound start, which has no document.
    digest: Option<Digest>,
    /// How deep the engine's stages read a request's block hashes.
    depth: HashDepth,
    /// Whether no stage of the engine reads those hashes itself
    /// ([`Self::hashes_named_alone`]).
    named_alone: bool,
    /// The document's `view.max_block_hashes`: no read is past it.
    max_block_hashes: u32,
    /// The view's block size: a request that names none is counted by it.
    block_size: u32,
    /// The index settings a model takes where the in-process engine reads
    /// a pool's table.
    #[cfg(feature = "in-process")]
    index: BTreeMap<String, IndexSpec>,
    pairing: Pairing,
    /// The document's `plans.pd-pair.decode`.
    decode_pick: DecodePick,
    lists: ViewLists,
    /// The document's `view.headers`, in their order
    /// ([`AppliedSpec::view_headers`]).
    view_headers: Box<[Box<str>]>,
    /// Each binding's routing key sources, by the binding's place in the
    /// document; none where a binding names none
    /// ([`AppliedSpec::key_sources_for`]).
    key_sources: Box<[Option<KeySources>]>,
    /// Whether a binding names a target source
    /// ([`AppliedSpec::names_target`]).
    names_target: bool,
    /// What the bindings say of the overload switch, taken together
    /// ([`AppliedSpec::overload_switch`]).
    overload_switch: OverloadSwitch,
    /// The document's signals, resolved, in column order
    /// ([`AppliedSpec::signal_columns`]).
    signal_columns: Arc<[SignalColumn]>,
    /// A start's document waits for the first lead of this store's
    /// replica ([`AppliedSpec::mark_document_awaited`]): set on the unbound start
    /// alone, and gone with it at the first publication.
    #[cfg(feature = "controller")]
    awaited: AtomicBool,
}

impl AppliedSpec {
    /// The generation `document` validates into ([`PolicySpec::validate`]).
    pub fn validate(document: &PolicySpec) -> Result<Self, SpecError> {
        Self::validate_with(document, &[])
    }

    /// [`AppliedSpec::validate`] with the host's own denied header names,
    /// as [`PolicySpec::validate_with`] takes them.
    pub fn validate_with(
        document: &PolicySpec,
        denied_headers: &[&str],
    ) -> Result<Self, SpecError> {
        Self::validate_in(document, &STOCK, denied_headers)
    }

    /// [`AppliedSpec::validate_with`] over the stages `catalog` holds,
    /// including stages the host registers.
    ///
    /// The caller assigns the document's generation number.
    /// [`MemoryStore::apply`](crate::store::MemoryStore::apply) accepts a
    /// generation whose number is above the standing one.
    pub fn validate_in(
        document: &PolicySpec,
        catalog: &StageRegistry,
        denied_headers: &[&str],
    ) -> Result<Self, SpecError> {
        let built = document.built(catalog, denied_headers)?;
        Ok(Self::of(document, document.digest()?, built))
    }

    /// [`AppliedSpec::validate`] over the stages `catalog` holds, under a
    /// digest its caller took already: the operator door reads the
    /// document's digest against the standing one and hands it on, so an
    /// apply digests its document once.
    #[cfg(any(test, feature = "operator"))]
    pub(crate) fn validate_digested(
        document: &PolicySpec,
        digest: Digest,
        catalog: &StageRegistry,
        denied_headers: &[&str],
    ) -> Result<Self, SpecError> {
        let built = document.built(catalog, denied_headers)?;
        Ok(Self::of(document, digest, built))
    }

    /// The generation of `document` under its `digest`: the engine it
    /// validated into, with the depth of that engine's stages.
    fn of(
        document: &PolicySpec,
        digest: Digest,
        (engine, depth, signal_columns): (Engine, HashDepths, Arc<[SignalColumn]>),
    ) -> Self {
        let sources = |binding| KeySources::of_binding(binding, &document.view);
        let key_sources: Box<[_]> = document.bindings.iter().map(sources).collect();
        Self {
            engine,
            #[cfg(feature = "in-process")]
            built_from: BuiltFrom {
                profiles: document.profiles.clone(),
                view: document.view.clone(),
                signals: document.signals.clone(),
            },
            digest: Some(digest),
            depth: depth.read,
            named_alone: depth.by_stage == HashDepth::Unread,
            signal_columns,
            max_block_hashes: document.view.max_block_hashes,
            block_size: Gates::of(None, None, document.view.block_size).block_size,
            #[cfg(feature = "in-process")]
            index: document.index.clone(),
            pairing: document.plans.pd_pair.pairing,
            decode_pick: document.plans.pd_pair.decode,
            lists: ViewLists::of(&document.view),
            view_headers: document
                .view
                .headers
                .iter()
                .map(|name| name.as_str().into())
                .collect(),
            names_target: key_sources.iter().flatten().any(KeySources::names_target),
            overload_switch: OverloadSwitch::of(&document.bindings),
            key_sources,
            #[cfg(feature = "controller")]
            awaited: AtomicBool::new(false),
        }
    }

    /// The generation a store holds before any document is applied: number
    /// 0, an engine with no profile and no binding, which declines every
    /// decision by name as an engine declines a model no binding covers,
    /// and the pair plan's rules a document that names none has.
    #[must_use]
    pub fn unbound() -> Self {
        Self {
            engine: Engine::new(Vec::new(), Vec::new(), 0, Needs::default()),
            #[cfg(feature = "in-process")]
            built_from: BuiltFrom::default(),
            digest: None,
            depth: HashDepth::Unread,
            named_alone: true,
            signal_columns: Arc::default(),
            max_block_hashes: ViewSpec::default().max_block_hashes,
            block_size: Gates::default().block_size,
            #[cfg(feature = "in-process")]
            index: BTreeMap::new(),
            pairing: Pairing::default(),
            decode_pick: DecodePick::default(),
            lists: ViewLists::default(),
            view_headers: Box::default(),
            key_sources: Box::default(),
            names_target: false,
            overload_switch: OverloadSwitch::On,
            #[cfg(feature = "controller")]
            awaited: AtomicBool::new(false),
        }
    }

    /// The document's signals, resolved, in column order: a signal's place
    /// here is its column's number. What a stage's build answered from
    /// ([`StageBuild::signal_column`](crate::stages::StageBuild::signal_column)),
    /// and the one reading of the names for whoever makes the cells: the
    /// engine over worker lists, or a host that builds its own.
    /// Empty for a document that declares none and for the unbound start.
    pub fn signal_columns(&self) -> &[SignalColumn] {
        &self.signal_columns
    }

    /// [`Self::signal_columns`] as the list is shared: what a worker list's
    /// records keep for a report's conversion.
    #[cfg(feature = "in-process")]
    pub(crate) fn shared_signal_columns(&self) -> Arc<[SignalColumn]> {
        Arc::clone(&self.signal_columns)
    }

    /// Marks the unbound start of a replica whose start's document waits
    /// for its first lead. Under the mark a request is no decision, and
    /// its answer is `Busy` with no reason: the unbound start of a store
    /// with no document sheds it by the binding, which would say here that
    /// a document refuses what no document has read. The mark is no
    /// generation: it takes no number and is told to no standby, and it
    /// goes with the unbound start at the first publication.
    #[cfg(feature = "controller")]
    pub(crate) fn mark_document_awaited(&self) {
        self.awaited.store(true, atomic::Ordering::Relaxed);
    }

    /// Whether a start's document waits for this generation's place
    /// ([`AppliedSpec::mark_document_awaited`]).
    #[cfg(feature = "controller")]
    pub(crate) fn awaits_document(&self) -> bool {
        self.awaited.load(atomic::Ordering::Relaxed)
    }

    /// The generation's number: the engine's own.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.engine.generation()
    }

    /// The digest of the document this generation was validated from
    /// ([`PolicySpec::digest`]); `None` for the unbound start, which has no
    /// document.
    #[must_use]
    pub fn digest(&self) -> Option<Digest> {
        self.digest
    }

    /// The engine a decision under this generation runs.
    #[must_use]
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// The document's engine reads and writes `table` from here on
    /// ([`Engine::share_sessions`]): the session table of an engine object
    /// that outlives the document.
    #[cfg(feature = "in-process")]
    pub(crate) fn share_sessions(&mut self, table: Arc<SessionTable>) {
        self.engine.share_sessions(table);
    }

    /// Keeps `standing`'s stage objects, and with them what the stages
    /// hold, for each profile whose name and text did not change
    /// ([`Engine::keep_from`]). A profile's build reads three blocks of its
    /// spec: its own, the spec's `view` and its `signals`. So a new spec
    /// that changes `signals` builds every profile anew, and so does one
    /// that changes `view`, with one exception.
    ///
    /// The exception is a view that differs in its header list alone
    /// ([`ViewSpec::same_but_for_headers`]). No stage is built from that
    /// list: a factory is handed none, and the validator's own stages read
    /// four other values of the view. A stage that reads a request's header
    /// reads it by a place its host wrote, under the list of the spec its
    /// decision runs under. So a kept stage reads what one built anew
    /// would.
    ///
    /// The engine keeps nothing across two stage tables that may build
    /// another stage under one name.
    #[cfg(feature = "in-process")]
    pub(crate) fn keep_from(&mut self, standing: &Self) {
        let (new, old) = (&self.built_from, &standing.built_from);
        if !new.view.same_but_for_headers(&old.view) || new.signals != old.signals {
            return;
        }
        let same = |name: &str| {
            let block = new.profiles.get(name);
            block.is_some() && block == old.profiles.get(name)
        };
        self.engine.keep_from(&standing.engine, same);
    }

    /// The document's block size, in tokens: the size a request's blocks
    /// are counted by where neither the request nor a record says one.
    #[must_use]
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// The most specific matching index settings, or the default.
    #[cfg(feature = "in-process")]
    pub(crate) fn index_for(&self, model: &str) -> IndexSpec {
        index_matching(&self.index, model)
            .map(|(_, spec)| spec.clone())
            .unwrap_or_default()
    }

    /// The most prefix block hashes of one request a decision under this
    /// generation reads: the document's `view.max_block_hashes`. What a
    /// decision reads of a request is [`Self::block_hashes_read`], this
    /// many at most.
    #[must_use]
    pub fn max_block_hashes(&self) -> usize {
        usize::try_from(self.max_block_hashes).unwrap_or(usize::MAX)
    }

    /// How many of a request's prefix block hashes a decision under this
    /// generation reads, for a request counted in `kind` by blocks of
    /// `block_size`, `0` for the view's size. The generation's depth is in
    /// tokens, the deepest `prefix_tokens` of its consistent-hash stages,
    /// and the blocks are the request's own, counted as the key counts
    /// them. A stage that declares the request's hashes and names no depth
    /// reads to the cap ([`Self::max_block_hashes`]), which no depth is
    /// past, and a generation with no such stage reads none. A kind no
    /// stage of the generation declares is read by none
    /// ([`Needs::token_hashes`], [`Needs::text_hashes`]): `0`, whatever the
    /// other kind's depth. A host hashes that many blocks of a request and
    /// no more, and hands the engine the first that many of a longer list.
    #[must_use]
    pub fn block_hashes_read(&self, kind: HashKind, block_size: u32) -> usize {
        let needs = self.needs();
        let declared = match kind {
            HashKind::Token => needs.token_hashes,
            HashKind::Text => needs.text_hashes,
        };
        if !declared {
            return 0;
        }
        let blocks = match self.depth {
            HashDepth::Unread => 0,
            HashDepth::Tokens(tokens) => {
                let block_size = if block_size > 0 {
                    block_size
                } else {
                    self.block_size
                };
                blocks_within(tokens, kind, block_size).min(self.max_block_hashes)
            }
            HashDepth::Whole => self.max_block_hashes,
        };
        usize::try_from(blocks).unwrap_or(usize::MAX)
    }

    /// Whether no stage reads a request's block hashes itself under this
    /// generation: the lookup that names the request's holders reads them
    /// where the prefix-affinity stage declares them, and nothing does
    /// where no stage declares any.
    /// The engine's own call then hashes a prompt where that lookup reads
    /// the list, and not for a request it decides inside a lent token tree.
    /// A request that call makes of a prompt's token ids with no block
    /// hashed is sized as the one it makes with its list (`HashDepths`).
    #[cfg_attr(
        not(feature = "in-process"),
        expect(dead_code, reason = "the engine's own placing call reads it")
    )]
    pub(crate) const fn hashes_named_alone(&self) -> bool {
        self.named_alone
    }

    /// What the host must feed under this generation
    /// ([`Engine::needs`]).
    #[must_use]
    pub fn needs(&self) -> Needs {
        self.engine.needs()
    }

    /// The sources a request of `phase` for `model` takes its routing key
    /// from: those of the first binding that matches it, a pair request's
    /// by its prefill lookup. The host reads the request by them
    /// ([`KeySources::derive`]) and hands the engine the one key
    /// ([`RequestFacts::session_key`](crate::engine::RequestFacts::session_key)).
    /// `None` where no binding matches or the binding names no source: the
    /// request carries no key.
    #[must_use]
    pub fn key_sources_for(&self, model: &str, phase: Phase) -> Option<&KeySources> {
        let at = self.engine.binding_position(model, phase)?;
        self.key_sources.get(usize::try_from(at).ok()?)?.as_ref()
    }

    /// Whether a binding of this generation takes a routing key from the
    /// request's own id ([`KeySources::reads_rid`]). A host that learns
    /// which binding a request meets from its body alone, where the model
    /// stands, keeps every request's body readable while this generation
    /// stands. A binding no request reaches counts too.
    #[must_use]
    pub fn reads_request_id(&self) -> bool {
        let mut sources = self.key_sources.iter().flatten();
        sources.any(KeySources::reads_rid)
    }

    /// Whether a binding of this generation names a target source
    /// ([`KeySourceSpec::target`]), decided when the document was applied.
    /// A request under a document that names none pays this one test, and
    /// no header of it is read for a target.
    #[must_use]
    pub fn names_target(&self) -> bool {
        self.names_target
    }

    /// What the bindings of this generation say of the overload switch,
    /// taken together ([`OverloadSwitch`]), decided when the document was
    /// applied. A host that builds its workers with a default set of
    /// thresholds reads it where it builds that default. The unbound start
    /// says on, as a document with no binding does.
    #[must_use]
    pub fn overload_switch(&self) -> OverloadSwitch {
        self.overload_switch
    }

    /// The mode a model's pair classes are derived under while this
    /// generation stands: the document's `plans.pd-pair.pairing`.
    #[must_use]
    pub fn pairing(&self) -> Pairing {
        self.pairing
    }

    /// When a pair's decode side is picked while this generation stands:
    /// the document's `plans.pd-pair.decode`. A host whose route sends a
    /// pair's decode behind its prefill states
    /// [`Constraints::decode_later`](crate::engine::view::Constraints::decode_later)
    /// under [`DecodePick::AfterPrefill`], and asks the decode again when
    /// the prefill has answered.
    #[must_use]
    pub fn decode_pick(&self) -> DecodePick {
        self.decode_pick
    }

    /// The lists this generation declares: a worker's candidate row is
    /// written under them, and a store counts the members outside them.
    #[must_use]
    pub fn lists(&self) -> &ViewLists {
        &self.lists
    }

    /// The request headers a stage may read (`view.headers`), in the
    /// document's order. A host states each one a request carries as a
    /// [`Header`](crate::engine::view::Header), which names its header by
    /// its place here.
    pub fn view_headers(&self) -> impl Iterator<Item = &str> {
        self.view_headers.iter().map(|name| &**name)
    }
}
