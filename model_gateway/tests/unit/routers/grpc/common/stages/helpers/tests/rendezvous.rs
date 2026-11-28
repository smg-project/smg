//! The bootstrap rendezvous a disaggregated request carries: the operator's wins and the engine's
//! fills the rest; the room follows a dp-aware prefill worker's rank.

use super::*;

/// What the operator registered wins, field by field; what the engine
/// declared fills the rest; the default port is the last resort.
#[test]
fn the_operators_rendezvous_wins_and_the_engines_fills_the_rest() {
    let declared = Some(("10.0.0.7", 31755));
    assert_eq!(
        bootstrap_rendezvous("", None, declared),
        ("10.0.0.7", 31755)
    );
    assert_eq!(
        bootstrap_rendezvous("prefill-0", Some(9000), declared),
        ("prefill-0", 9000)
    );
    assert_eq!(
        bootstrap_rendezvous("prefill-0", None, declared),
        ("prefill-0", 31755)
    );
    assert_eq!(
        bootstrap_rendezvous("", Some(9000), declared),
        ("10.0.0.7", 9000)
    );
    assert_eq!(
        bootstrap_rendezvous("prefill-0", None, None),
        ("prefill-0", DEFAULT_BOOTSTRAP_PORT)
    );
}

/// On each runtime whose request carries a room, a dp-aware prefill worker's
/// room is congruent to its rank inside the wire's width; a prefill worker
/// of one rank, of no DP size, or of a DP size the room cannot carry gets a
/// free room.
#[test]
fn a_minted_room_follows_the_prefill_rank_only_on_a_multi_rank_worker() {
    use smg_worker::identity::basic::{RuntimeType, WorkerType};

    use crate::routers::grpc::backend_client::BasicWorkerBuilder;

    let rooms = |runtime_type: RuntimeType, dp: Option<(usize, usize)>| -> Vec<i64> {
        let worker = |worker_type: WorkerType| -> Arc<dyn Worker> {
            let builder = BasicWorkerBuilder::new("grpc://worker:30000")
                .worker_type(worker_type)
                .runtime_type(runtime_type);
            Arc::new(
                match dp {
                    Some((rank, size)) => builder.dp_config(rank, size),
                    None => builder,
                }
                .build(),
            )
        };
        let workers = WorkerSelection::Disaggregated {
            encode_assignments: None,
            prefill: worker(WorkerType::Prefill),
            decode: worker(WorkerType::Decode).into(),
            runtime_type,
        };
        (0..64)
            .map(|_| {
                let mut request = match runtime_type {
                    RuntimeType::Sglang => ProtoGenerateRequest::Sglang(Default::default()),
                    _ => ProtoGenerateRequest::TokenSpeed(Default::default()),
                };
                maybe_inject_pd_metadata(&mut request, &workers);
                maybe_inject_pd_rendezvous(&mut request, &workers);
                match request {
                    ProtoGenerateRequest::Sglang(request) => {
                        i64::from(request.disaggregated_params.unwrap().bootstrap_room)
                    }
                    ProtoGenerateRequest::TokenSpeed(request) => {
                        request.kv_bootstrap_info.unwrap().bootstrap_room
                    }
                    _ => panic!("the request carries no room"),
                }
            })
            .collect()
    };

    for (runtime_type, bound) in [
        (RuntimeType::Sglang, i64::from(i32::MAX)),
        (RuntimeType::TokenSpeed, i64::MAX),
    ] {
        for room in rooms(runtime_type, Some((1, 4))) {
            assert!(
                (0..bound).contains(&room) && room % 4 == 1,
                "{runtime_type:?}: {room}"
            );
        }
        for dp in [Some((0, 1)), None] {
            let residues: std::collections::HashSet<i64> = rooms(runtime_type, dp)
                .iter()
                .map(|room| room % 4)
                .collect();
            assert!(residues.len() > 1, "{runtime_type:?} {dp:?}: {residues:?}");
        }
        // The ranked draw's range is `0..bound - size`: `edge` is the last
        // size it is not empty for (one room, the rank itself), and from
        // `edge + 1` on the worker takes the free room.
        let edge = usize::try_from(bound - 1).unwrap();
        for room in rooms(runtime_type, Some((1, edge))) {
            assert_eq!(room, 1, "{runtime_type:?} {edge}");
        }
        for size in [edge + 1, usize::MAX] {
            let free: std::collections::HashSet<i64> =
                rooms(runtime_type, Some((1, size))).into_iter().collect();
            assert!(
                free.len() > 1 && free.iter().all(|room| (0..bound).contains(room)),
                "{runtime_type:?} {size}: {free:?}"
            );
        }
    }
}
