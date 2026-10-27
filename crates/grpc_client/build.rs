fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Rebuild triggers
    println!("cargo:rerun-if-changed=proto/common.proto");
    println!("cargo:rerun-if-changed=proto/sglang_scheduler.proto");
    println!("cargo:rerun-if-changed=proto/tokenspeed_scheduler.proto");
    println!("cargo:rerun-if-changed=proto/tokenspeed_encoder.proto");
    println!("cargo:rerun-if-changed=proto/vllm_engine.proto");
    println!("cargo:rerun-if-changed=proto/trtllm_service.proto");
    println!("cargo:rerun-if-changed=proto/mlx_engine.proto");

    // The generated clients and servers share the message types and the
    // codec (`crate::codec`), with one difference fixed here: a client's
    // codec starts a request stream's encode buffer at the size a text
    // request fits, a server's keeps the default for a response stream.
    // tonic-build takes one codec path per pass, so each service proto set
    // is compiled twice: the server pass emits the messages and the servers
    // into OUT_DIR, the client pass emits the clients alone into
    // OUT_DIR/client, with every message package mapped to the module the
    // server pass fills (prost generates nothing for a mapped package).
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR")?);
    let client_dir = out_dir.join("client");
    std::fs::create_dir_all(&client_dir)?;

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
        .build_client(false)
        .extern_path(
            ".tokenspeed.grpc.scheduler",
            "crate::tokenspeed_scheduler::tokenspeed_proto",
        )
        // The sized codec at tonic's default start size (see
        // `crate::codec::ServerCodec`): a server encodes the responses.
        .codec_path("crate::codec::ServerCodec")
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&["proto/tokenspeed_encoder.proto"], &["proto"])?;
    // The encoder's client. Its Encode RPC carries the same pixel tensor as
    // the scheduler's Generate, in the request the client encodes: the same
    // sized codec (see `crate::codec::ClientCodec`).
    tonic_prost_build::configure()
        .build_server(false)
        .build_client(true)
        .out_dir(&client_dir)
        .extern_path(".smg.grpc.common", "crate::common_proto")
        .extern_path(
            ".tokenspeed.grpc.scheduler",
            "crate::tokenspeed_scheduler::tokenspeed_proto",
        )
        .extern_path(
            ".tokenspeed.grpc.encoder",
            "crate::tokenspeed_encoder::tokenspeed_encoder_proto",
        )
        .codec_path("crate::codec::ClientCodec")
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&["proto/tokenspeed_encoder.proto"], &["proto"])?;

    // Pass 3: compile engine protos, referencing common types via extern_path
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(false)
        .extern_path(".smg.grpc.common", "crate::common_proto")
        // The inline tensor payloads are `Bytes`: a multi-megabyte pixel
        // tensor is then shared, not copied, by every clone of the request
        // that carries it (a retry attempt's, a disaggregated side's). The
        // wire encoding is the same `bytes` field.
        .bytes(".tokenspeed.grpc.scheduler.TensorData.inline")
        .bytes(".vllm.grpc.engine.TensorData.inline")
        .bytes(".sglang.grpc.scheduler.TensorData.data")
        // The sized codec at tonic's default start size (see
        // `crate::codec::ServerCodec`): a server encodes the responses.
        .codec_path("crate::codec::ServerCodec")
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
    // The engine clients. Each request is encoded into a buffer reserved to
    // its length once (see `crate::codec::ClientCodec`), not into one that
    // doubles under a tensor.
    tonic_prost_build::configure()
        .build_server(false)
        .build_client(true)
        .out_dir(&client_dir)
        .extern_path(".smg.grpc.common", "crate::common_proto")
        .extern_path(".sglang.grpc.scheduler", "crate::sglang_scheduler::proto")
        .extern_path(".vllm.grpc.engine", "crate::vllm_engine::proto")
        .extern_path(".trtllm", "crate::trtllm_service::proto")
        .extern_path(".mlx.grpc.engine", "crate::mlx_engine::proto")
        .extern_path(
            ".tokenspeed.grpc.scheduler",
            "crate::tokenspeed_scheduler::tokenspeed_proto",
        )
        .codec_path("crate::codec::ClientCodec")
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
