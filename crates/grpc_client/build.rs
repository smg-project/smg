fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Rebuild triggers
    println!("cargo:rerun-if-changed=proto/common.proto");
    println!("cargo:rerun-if-changed=proto/sglang_scheduler.proto");
    println!("cargo:rerun-if-changed=proto/sglang_runtime.proto");
    println!("cargo:rerun-if-changed=proto/tokenspeed_scheduler.proto");
    println!("cargo:rerun-if-changed=proto/tokenspeed_encoder.proto");
    println!("cargo:rerun-if-changed=proto/vllm_engine.proto");
    println!("cargo:rerun-if-changed=proto/trtllm_service.proto");
    println!("cargo:rerun-if-changed=proto/mlx_engine.proto");
    println!("cargo:rerun-if-changed=proto/worker_control.proto");
    println!("cargo:rerun-if-changed=proto/worker_inference.proto");

    // Pass 1: compile shared message types (no gRPC service generation)
    tonic_prost_build::configure()
        .build_server(false)
        .build_client(false)
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&["proto/common.proto"], &["proto"])?;

    // Pass 2: the EPD encoder proto reuses tokenspeed scheduler message types
    // (MultimodalInputs/TensorData). Compile it with the scheduler package
    // mapped to its already-defined module so the encoder references those
    // types via the crate path instead of a (nonexistent) sibling module. Run
    // this BEFORE the scheduler pass so the full scheduler output (Pass 3) wins
    // over the extern stub this pass emits for the imported package.
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .extern_path(
            ".tokenspeed.grpc.scheduler",
            "crate::tokenspeed_scheduler::tokenspeed_proto",
        )
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&["proto/tokenspeed_encoder.proto"], &["proto"])?;

    // Pass 3: compile engine protos, referencing common types via extern_path
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .extern_path(".smg.grpc.common", "crate::common_proto")
        .type_attribute("GetModelInfoResponse", "#[derive(serde::Serialize)]")
        // Some ServerInfo protos contain prost_types::{Struct, Timestamp};
        // those are handled separately at the wrapper layer.
        .type_attribute(
            "vllm.grpc.engine.GetServerInfoResponse",
            "#[derive(serde::Serialize)]",
        )
        .type_attribute(
            "trtllm.GetServerInfoResponse",
            "#[derive(serde::Serialize)]",
        )
        .type_attribute(
            "mlx.grpc.engine.GetServerInfoResponse",
            "#[derive(serde::Serialize)]",
        )
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(
            &[
                "proto/sglang_scheduler.proto",
                "proto/sglang_runtime.proto",
                "proto/vllm_engine.proto",
                "proto/trtllm_service.proto",
                "proto/mlx_engine.proto",
                "proto/tokenspeed_scheduler.proto",
                "proto/worker_control.proto",
                "proto/worker_inference.proto",
            ],
            &["proto"],
        )?;

    Ok(())
}
