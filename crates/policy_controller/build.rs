//! The controller's protocol as Rust: the messages, the service and its
//! client, generated from the two contract files when the `controller`
//! feature is on. The engine alone runs nothing here and needs no `protoc`.

#[cfg(not(feature = "controller"))]
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
}

/// One step for the server and the client, so the two cannot drift. The
/// engine wire's package (`smg.grpc.common`: the load record and the event
/// batch a relay forwards unchanged) is the types the gRPC client crate
/// re-exports from the worker crate, never a second copy.
#[cfg(feature = "controller")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    const CONTRACTS: [&str; 2] = [
        "contracts/policy_plugin.proto",
        "contracts/policy_controller.proto",
    ];
    // The worker crate's proto directory, by its path in the repository: this
    // crate is built from a checkout, in the workspace or imported by git.
    const ENGINE_WIRE: &str = "../worker/proto";
    println!("cargo:rerun-if-changed=build.rs");
    for contract in CONTRACTS {
        println!("cargo:rerun-if-changed={contract}");
    }
    println!("cargo:rerun-if-changed={ENGINE_WIRE}/common.proto");
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .extern_path(".smg.grpc.common", "::smg_grpc_client::common_proto")
        // A snapshot's part is one case of a batch, and the generation it
        // states one field of a part: boxed, a batch of events carries one
        // pointer for it and not the generation's own size.
        .boxed(".smg.policy.v1.ReplicationSnapshot.generation")
        // proto3 `optional` needs the flag below protoc 3.15, the oldest the
        // repository's build scripts run under.
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&CONTRACTS, &["contracts", ENGINE_WIRE])?;
    Ok(())
}
