#!/bin/bash
set -e

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_SOURCE_DIR="$SCRIPT_DIR/../proto"
GO_PROTO_DIR="$SCRIPT_DIR/proto"
OUTPUT_DIR="$SCRIPT_DIR/generated"

# Ensure directories exist
rm -rf "$GO_PROTO_DIR" "$OUTPUT_DIR"
mkdir -p "$GO_PROTO_DIR"
mkdir -p "$OUTPUT_DIR/common"
mkdir -p "$OUTPUT_DIR/sglang_encoder"
mkdir -p "$OUTPUT_DIR/sglang_scheduler"
mkdir -p "$OUTPUT_DIR/trtllm"
mkdir -p "$OUTPUT_DIR/vllm"

# Copy proto files
echo "Copying proto files..."
cp "$PROTO_SOURCE_DIR"/*.proto "$GO_PROTO_DIR"/
# The shared messages' file is the worker crate's.
cp "$SCRIPT_DIR/../../worker/proto/common.proto" "$GO_PROTO_DIR"/

# Find tools in PATH or GOPATH
PROTOC_GEN_GO=$(command -v protoc-gen-go || echo "$(go env GOPATH)/bin/protoc-gen-go")
PROTOC_GEN_GO_GRPC=$(command -v protoc-gen-go-grpc || echo "$(go env GOPATH)/bin/protoc-gen-go-grpc")

if [ ! -x "$PROTOC_GEN_GO" ]; then
  echo "Error: protoc-gen-go not found. Please install with: go install google.golang.org/protobuf/cmd/protoc-gen-go@latest" >&2
  exit 1
fi

if [ ! -x "$PROTOC_GEN_GO_GRPC" ]; then
  echo "Error: protoc-gen-go-grpc not found. Please install with: go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest" >&2
  exit 1
fi

# Common mappings for imports
MAPPINGS="--go_opt=Mcommon.proto=github.com/smg-project/smg/crates/grpc_client/go/generated/common"
MAPPINGS_GRPC="--go-grpc_opt=Mcommon.proto=github.com/smg-project/smg/crates/grpc_client/go/generated/common"

echo "Generating Go code..."

cd "$GO_PROTO_DIR"

generate_service() {
  local service_name="$1"
  local proto_file="$2"
  local output_dir="$OUTPUT_DIR/$service_name"

  echo "Generating for $service_name..."

  protoc \
    --plugin=protoc-gen-go="$PROTOC_GEN_GO" \
    --plugin=protoc-gen-go-grpc="$PROTOC_GEN_GO_GRPC" \
    --proto_path=. \
    --go_out="$output_dir" \
    --go_opt=paths=source_relative \
    "$MAPPINGS" \
    --go_opt="M${proto_file}=github.com/smg-project/smg/crates/grpc_client/go/generated/${service_name}" \
    --go-grpc_out="$output_dir" \
    --go-grpc_opt=paths=source_relative \
    "$MAPPINGS_GRPC" \
    --go-grpc_opt="M${proto_file}=github.com/smg-project/smg/crates/grpc_client/go/generated/${service_name}" \
    "$proto_file"
}

# Generate all services
generate_service "common" "common.proto"
generate_service "sglang_encoder" "sglang_encoder.proto"
generate_service "sglang_scheduler" "sglang_scheduler.proto"
generate_service "trtllm" "trtllm_service.proto"
generate_service "vllm" "vllm_engine.proto"

echo "Generation complete!"
