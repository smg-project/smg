//! The load call: before the store shows a world it tells the world's
//! engine each rank's report, where a stage of that engine keeps a number
//! of them.

/// The cases that hold a stage of their own: a generation over a host's
/// catalog is made at the controller's door.
#[cfg(feature = "controller")]
mod held {
    use std::{
        sync::{Arc, Weak},
        time::{Duration, Instant},
    };

    use parking_lot::Mutex;

    use super::super::{ask, load, member, MODEL};
    use crate::{
        builtins::target_of,
        counts::Slot,
        decision::Targets,
        framework::{Picker, Scorer, TieBand},
        lease::{Lifecycle, Signal},
        spec::BindingSpec,
        store::{world::ViewBuffers, Kinds, LoadSink, LoadSource, MemoryStore},
        view::{Candidate, RequestFacts, Role, WorkerId},
        AppliedSpec, Catalog, PolicySpec,
    };

    /// One word a stage took: the worker, the requests its report says
    /// run, whether the word names a slot, the report's sample instant,
    /// and the sample instant of the report the standing world showed for
    /// that worker when the word came, zero for none.
    type Word = (WorkerId, u32, bool, u64, u64);

    /// What a stage was told, in the order it was told.
    type Told = Arc<Mutex<Vec<Word>>>;

    /// A scorer that scores every row alike and writes down each report it
    /// is told.
    #[derive(Debug)]
    struct Keeper {
        name: &'static str,
        keeps: bool,
        told: Told,
        store: Weak<MemoryStore>,
    }

    impl Lifecycle for Keeper {
        fn on_signal(&self, signal: &Signal) {
            let Signal::Load {
                worker,
                row,
                slot,
                sampled_at_micros,
                ..
            } = *signal
            else {
                return;
            };
            let store = self.store.upgrade().expect("the store stands");
            // Under the load lock: the worlds' words come in the worlds'
            // order.
            assert!(!store.load_lock_free(), "told outside the load lock");
            let shown = shown(&store, worker);
            let named = slot != Slot::NONE;
            self.told
                .lock()
                .push((worker, row.running, named, sampled_at_micros, shown));
        }

        fn keeps_loads(&self) -> bool {
            self.keeps
        }
    }

    impl Scorer for Keeper {
        fn name(&self) -> &'static str {
            self.name
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            out.fill(0.0);
            TieBand::NORMALIZED
        }
    }

    impl Picker for Keeper {
        fn name(&self) -> &'static str {
            self.name
        }

        /// The first row handed, whatever it scores.
        fn pick(
            &self,
            _: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            _: &[f32],
            _: TieBand,
            out: &mut Targets,
        ) {
            if let Some(&row) = kept.first() {
                out.push(target_of(&candidates[row as usize]));
            }
        }
    }

    /// The sample instant of the report the standing world shows for
    /// `worker`: zero where it shows none.
    fn shown(store: &MemoryStore, worker: WorkerId) -> u64 {
        let mut out = ViewBuffers::default();
        let world = store.world();
        if world
            .build(MODEL, Role::Single, &ask(Instant::now()), &mut out)
            .is_none()
        {
            return 0;
        }
        let row = out.candidates.iter().find(|row| row.id == worker);
        row.map_or(0, |row| row.load_sampled_at_micros)
    }

    /// Generation `number` of one profile for every model, scored by a
    /// keeper that says `keeps` of the reports.
    fn generation(number: u64, store: &Arc<MemoryStore>, told: &Told, keeps: bool) -> AppliedSpec {
        generation_of(number, store, &[("keeper", told, keeps)], None)
    }

    /// Generation `number` of one profile for every model, scored by one
    /// keeper a name: each writes down what it is told and says of the
    /// reports what its entry says. A keeper named as `picks` is the
    /// profile's picker; with none the built-in best picks.
    fn generation_of(
        number: u64,
        store: &Arc<MemoryStore>,
        stages: &[(&'static str, &Told, bool)],
        picks: Option<(&'static str, &Told, bool)>,
    ) -> AppliedSpec {
        let mut catalog = Catalog::builtin();
        for &(name, told, keeps) in stages {
            let (told, store) = (Arc::clone(told), Arc::downgrade(store));
            catalog
                .scorer(name, move || {
                    Box::new(Keeper {
                        name,
                        keeps,
                        told: Arc::clone(&told),
                        store: Weak::clone(&store),
                    })
                })
                .expect("a free name");
        }
        let pick = picks.map_or("best", |(name, told, keeps)| {
            let (told, store) = (Arc::clone(told), Arc::downgrade(store));
            catalog
                .picker(name, move || {
                    Box::new(Keeper {
                        name,
                        keeps,
                        told: Arc::clone(&told),
                        store: Weak::clone(&store),
                    })
                })
                .expect("a free name");
            name
        });
        let names: Vec<&str> = stages.iter().map(|&(name, ..)| name).collect();
        let score = serde_json::to_string(&names).expect("the names");
        let mut document: PolicySpec = serde_json::from_str(&format!(
            r#"{{"apiVersion":"smg.policy/v1","profiles":{{"p":{{"score":{score},"pick":"{pick}"}}}}}}"#
        ))
        .expect("a document");
        document.generation = number;
        document.bindings = vec![BindingSpec {
            profile: Some("p".to_string()),
            ..BindingSpec::default()
        }];
        let digest = document.digest().expect("a digest");
        AppliedSpec::validate_digested(&document, digest, &catalog).expect("the document validates")
    }

    /// A stage that keeps the reports is told each rank's report with the
    /// rank's slot and the report's sample instant, under the load lock
    /// and before the world that holds the report is shown: the standing
    /// world still shows the report before it. A report told once is told
    /// again at each later world, the same word, and a generation applied
    /// over reports on file is told them at its apply. A stage that keeps
    /// none is told nothing.
    #[test]
    fn a_world_s_reports_are_told_before_the_world_is_shown() {
        let store = Arc::new(MemoryStore::new());
        let told = Told::default();
        store
            .apply(generation(1, &store, &told, true))
            .expect("applied");
        let id = member(&store, "http://a:8000");
        assert!(told.lock().is_empty(), "a member with no report");
        let t0 = Instant::now();
        store.record(id, 0, load(3, 0), t0, LoadSource::Stream);
        let first = shown(&store, id);
        assert_ne!(first, 0, "the report is shown");
        assert_eq!(
            told.lock().first(),
            Some(&(id, 3, true, first, 0)),
            "told while the world before it stood"
        );
        store.record(
            id,
            0,
            load(5, 0),
            t0 + Duration::from_millis(40),
            LoadSource::Stream,
        );
        let second = shown(&store, id);
        assert!(second > first);
        let newer = told.lock().iter().find(|word| word.1 == 5).copied();
        assert_eq!(
            newer,
            Some((id, 5, true, second, first)),
            "the newer report, told under the older world"
        );
        // A world that holds a report already told tells it again: a
        // member joins, and its build's world is told the one report.
        member(&store, "http://b:8000");
        assert_eq!(told.lock().last(), Some(&(id, 5, true, second, second)));

        // A generation applied over the report on file is told it.
        let next = Told::default();
        store
            .apply(generation(2, &store, &next, true))
            .expect("applied");
        assert_eq!(next.lock().last(), Some(&(id, 5, true, second, second)));

        // A stage that keeps no report is told none.
        let none = Told::default();
        store
            .apply(generation(3, &store, &none, false))
            .expect("applied");
        store.record(
            id,
            0,
            load(7, 0),
            t0 + Duration::from_millis(80),
            LoadSource::Stream,
        );
        assert!(none.lock().is_empty());
    }

    /// A restore replaces every row and shows its world at once: the
    /// standing generation's stage is told the reports the cut brings
    /// under the load lock, before that world is shown.
    #[test]
    fn a_restores_reports_are_told_before_its_world_is_shown() {
        let leader = MemoryStore::new();
        let standby = Arc::new(MemoryStore::new());
        let told = Told::default();
        standby
            .apply(generation(1, &standby, &told, true))
            .expect("applied");
        let id = member(&leader, "http://a:8000");
        leader.record(id, 0, load(3, 0), Instant::now(), LoadSource::Stream);
        assert!(told.lock().is_empty(), "nothing of the leader's yet");
        assert!(standby.restore_from(leader.cut(Kinds::All)));
        let restored = shown(&standby, id);
        assert_ne!(restored, 0, "the restore's world shows the report");
        assert_eq!(
            *told.lock(),
            [(id, 3, true, restored, 0)],
            "told once, while the world before the restore's stood"
        );
    }

    /// A world's reports go to the stage that keeps them and to no other
    /// stage of its engine: one that watches signals and keeps no report
    /// hears nothing of a publication, as before any stage kept one.
    #[test]
    fn a_stage_that_keeps_no_report_is_told_none_beside_one_that_does() {
        let store = Arc::new(MemoryStore::new());
        let (told, heard) = (Told::default(), Told::default());
        let stages = [("keeper", &told, true), ("watcher", &heard, false)];
        store
            .apply(generation_of(1, &store, &stages, None))
            .expect("applied");
        let id = member(&store, "http://a:8000");
        store.record(id, 0, load(3, 0), Instant::now(), LoadSource::Stream);
        assert_eq!(told.lock().len(), 1, "the keeper's one word");
        assert!(heard.lock().is_empty(), "a stage that keeps none heard");
    }

    /// A picker hears a world's reports by the same word as a scorer: one
    /// that keeps them is told each, and one that keeps none is told
    /// nothing, beside a scorer that keeps them.
    #[test]
    fn a_picker_is_told_a_world_s_reports_where_it_keeps_them() {
        for keeps in [true, false] {
            let store = Arc::new(MemoryStore::new());
            let (told, picked) = (Told::default(), Told::default());
            let stages = [("keeper", &told, true)];
            let picks = Some(("picks", &picked, keeps));
            store
                .apply(generation_of(1, &store, &stages, picks))
                .expect("applied");
            let id = member(&store, "http://a:8000");
            store.record(id, 0, load(3, 0), Instant::now(), LoadSource::Stream);
            assert_eq!(told.lock().len(), 1, "the scorer's one word");
            assert_eq!(picked.lock().len(), usize::from(keeps), "keeps: {keeps}");
        }
    }
}
