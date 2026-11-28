//! The resident memory of a serving controller, idle and with a thousand
//! workers in its store: a measurement, run by name.

use std::{io::Write, time::Duration};

use smg_grand_central::{
    engine::Role,
    service::controller::{Controller, State},
    store::{Registrant, WorkerFacts},
};
use tokio::sync::oneshot;

use crate::{config, net::vm_rss_kib};

/// `VmRSS` of this process, in KiB; `0` where `/proc` has no answer.
/// The Numbers table's resident rows: the controller serving idle, then with
/// a thousand workers in its store. A measurement, run by name with
/// `--ignored`; it prints and asserts only that the feed landed.
#[tokio::test]
#[ignore = "a measurement for the Numbers table, run by name"]
async fn rss_idle_and_with_a_thousand_workers_in_the_store() {
    let controller = Controller::bind(config()).await.expect("bind");
    let (store, holder) = (controller.store(), controller.workers());
    let mut readiness = controller.readiness();
    let (stop, stopped) = oneshot::channel::<()>();
    let serve = controller.serve(async {
        let _ = stopped.await;
    });
    let feed = async {
        assert_eq!(readiness.wait_for(State::Leader).await, Some(State::Leader));
        tokio::time::sleep(Duration::from_millis(200)).await;
        let idle = vm_rss_kib();
        for i in 0..1_000u32 {
            let key = format!("http://worker-{i}.fleet:8000");
            let stated = WorkerFacts {
                key: key.clone(),
                engine: key,
                incarnation: 1,
                role: Role::Single,
                models: vec!["model-a".to_owned()],
                dp_size: 1,
                max_running: Some(256),
                ..WorkerFacts::default()
            };
            let change = holder.upsert(&store, stated, Registrant::Control);
            change.expect("the store took the registration");
        }
        // The fleet arrived inside one window: its one build, asked here so
        // the figure holds the world a decision reads.
        store.publish_now();
        // A copy of the rows is resident at the sampling: the holder's cut.
        let cut = holder.checkpoint(&store);
        assert_eq!(cut.rows.len(), 1_000);
        let fed = vm_rss_kib();
        drop(cut);
        let mut out = std::io::stdout();
        writeln!(out, "RSS_IDLE_KIB {idle}").expect("stdout");
        writeln!(out, "RSS_1K_WORKERS_KIB {fed}").expect("stdout");
        stop.send(()).expect("the serve is listening for the stop");
    };
    let (served, ()) = tokio::join!(serve, feed);
    served.expect("a clean stop");
}
