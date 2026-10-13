fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Rebuild triggers
    println!("cargo:rerun-if-changed=proto/common.proto");
    println!("cargo:rerun-if-changed=proto/sglang_scheduler.proto");
    println!("cargo:rerun-if-changed=proto/tokenspeed_scheduler.proto");
    println!("cargo:rerun-if-changed=proto/tokenspeed_encoder.proto");
    println!("cargo:rerun-if-changed=proto/vllm_engine.proto");
    println!("cargo:rerun-if-changed=proto/trtllm_service.proto");
    println!("cargo:rerun-if-changed=proto/mlx_engine.proto");

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
        // The encoder's Encode RPC carries the same pixel tensor as the
        // scheduler's Generate: the same sized codec (see `crate::codec`).
        .codec_path("crate::codec::SizedProstCodec")
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&["proto/tokenspeed_encoder.proto"], &["proto"])?;

    // Pass 3: compile engine protos, referencing common types via extern_path
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .extern_path(".smg.grpc.common", "crate::common_proto")
        // The inline tensor payloads are `Bytes`: a multi-megabyte pixel
        // tensor is then shared, not copied, by every clone of the request
        // that carries it (a retry attempt's, a disaggregated leg's). The
        // wire encoding is the same `bytes` field.
        .bytes(".tokenspeed.grpc.scheduler.TensorData.inline")
        .bytes(".vllm.grpc.engine.TensorData.inline")
        .bytes(".sglang.grpc.scheduler.TensorData.data")
        // Each message is encoded into a buffer reserved to its length once
        // (see `crate::codec`), not into one that doubles under a tensor.
        .codec_path("crate::codec::SizedProstCodec")
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
                "proto/vllm_engine.proto",
                "proto/trtllm_service.proto",
                "proto/mlx_engine.proto",
                "proto/tokenspeed_scheduler.proto",
            ],
            &["proto"],
        )?;

    Ok(())
}
