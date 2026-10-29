# Model Gateway Makefile
# Provides convenient shortcuts for common development tasks

# Python bindings directory
PYTHON_DIR := bindings/python

# OpenAPI Generator CLI wrapper version (pinned for reproducibility)
OPENAPI_GENERATOR_CLI_VERSION := 2.30.0

# Auto-detect CPU cores and cap at reasonable limit to avoid thread exhaustion
# Can be overridden: make python-dev JOBS=4
NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
JOBS ?= $(shell echo $$(($(NPROC) > 16 ? 16 : $(NPROC))))

# Check if sccache is available and set RUSTC_WRAPPER accordingly
SCCACHE := $(shell which sccache 2>/dev/null)
ifdef SCCACHE
    export RUSTC_WRAPPER := $(SCCACHE)
    $(info Using sccache for compilation caching)
else
    $(info sccache not found. Install it for faster builds: cargo install sccache)
endif

.PHONY: help build test clean docs check fmt opencv-deps dev-setup pre-commit setup-sccache sccache-stats sccache-clean sccache-stop \
        python-dev python-build python-build-release python-install python-clean python-test python-check \
        generate-openapi generate-python-types generate-java-types generate-clients \
        show-version bump-version check-versions \
        release-plan release-cut release-bump release-pick release-pick-pr release-status release-tag release-notes release-bump-main release-publish-plan release-check

help: ## Show this help message
	@echo "Model Gateway Development Commands"
	@echo "=================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

build: ## Build the project in release mode
	@echo "Building Shepherd Model Gateway..."
	@cargo build --release

test: ## Run all tests
	@echo "Running tests..."
	@cargo test

clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	@cargo clean

docs: ## Generate and open documentation
	@echo "Generating documentation..."
	@cargo doc --open

check: ## Run cargo check and clippy
	@echo "Running cargo check..."
	@cargo check
	@echo "Running clippy..."
	@cargo clippy --all-targets --all-features -- -D warnings

fmt: ## Format code with rustfmt
	@echo "Formatting code..."
	@rustup run nightly cargo fmt

opencv-deps: ## Install system OpenCV for the opencv-video feature (needed by check/--all-features)
	@bash scripts/install_opencv.sh

# Development workflow shortcuts
dev-setup: build test ## Set up development environment
	@echo "Development environment ready!"

pre-commit: fmt check test ## Run pre-commit checks
	@echo "Pre-commit checks passed!"

# sccache management targets
setup-sccache: ## Install and configure sccache
	@echo "Setting up sccache..."
	@./scripts/setup-sccache.sh

sccache-stats: ## Show sccache statistics
	@if [ -n "$(SCCACHE)" ]; then \
		echo "sccache statistics:"; \
		sccache -s; \
	else \
		echo "sccache not installed. Run 'make setup-sccache' to install it."; \
	fi

sccache-clean: ## Clear sccache cache
	@if [ -n "$(SCCACHE)" ]; then \
		echo "Clearing sccache cache..."; \
		sccache -C; \
		echo "sccache cache cleared"; \
	else \
		echo "sccache not installed"; \
	fi

sccache-stop: ## Stop the sccache server
	@if [ -n "$(SCCACHE)" ]; then \
		echo "Stopping sccache server..."; \
		sccache --stop-server || true; \
	else \
		echo "sccache not installed"; \
	fi

# Python bindings (maturin) targets
python-dev: ## Build Python bindings in development mode (fast, debug build)
	@echo "Building Python bindings in development mode (using $(JOBS) parallel jobs with sccache)..."
	@cd $(PYTHON_DIR) && CARGO_BUILD_JOBS=$(JOBS) maturin develop

python-build: ## Build Python wheel (release mode with vendored OpenSSL)
	@echo "Building Python wheel (release, vendored OpenSSL, using $(JOBS) parallel jobs with sccache)..."
	@cd $(PYTHON_DIR) && CARGO_BUILD_JOBS=$(JOBS) maturin build --release --out dist --features vendored-openssl

python-build-release: python-build ## Alias for python-build

python-install: python-build ## Build and install Python wheel
	@echo "Installing Python wheel..."
	@pip install --force-reinstall $(PYTHON_DIR)/dist/*.whl
	@echo "Python package installed!"

python-clean: ## Clean Python build artifacts
	@echo "Cleaning Python build artifacts..."
	@rm -rf $(PYTHON_DIR)/dist/
	@rm -rf $(PYTHON_DIR)/target/
	@rm -rf $(PYTHON_DIR)/smg.egg-info/
	@rm -rf $(PYTHON_DIR)/smg/__pycache__/
	@find $(PYTHON_DIR) -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find $(PYTHON_DIR) -name "*.pyc" -delete 2>/dev/null || true
	@echo "Python build artifacts cleaned!"

python-test: ## Run Python tests
	@echo "Running Python tests..."
	@pytest e2e_test/ -v

python-check: ## Check Python package with twine
	@echo "Checking Python package..."
	@cd $(PYTHON_DIR) && CARGO_BUILD_JOBS=$(JOBS) maturin build --release --out dist --features vendored-openssl
	@pip install twine 2>/dev/null || true
	@twine check $(PYTHON_DIR)/dist/*
	@echo "Python package check passed!"

# Client SDK code generation
generate-openapi: ## Generate OpenAPI spec from Rust protocol types
	@echo "Generating OpenAPI spec..."
	@mkdir -p clients/openapi
	@cargo run -p openapi-gen -- clients/openapi/smg-openapi.yaml

generate-python-types: generate-openapi ## Generate Python types from OpenAPI spec
	@echo "Generating Python types..."
	@uvx --from 'datamodel-code-generator==0.54.0' datamodel-codegen \
		--input clients/openapi/smg-openapi.yaml \
		--input-file-type openapi \
		--output clients/python/smg_client/types/_generated.py \
		--output-model-type pydantic_v2.BaseModel \
		--use-annotated \
		--field-constraints \
		--target-python-version 3.10 \
		--collapse-root-models \
		--use-standard-collections \
		--use-union-operator
	@echo "Post-processing: converting enums to str enums..."
	@sed -i.bak 's/class \(.*\)(Enum):/class \1(str, Enum):/' clients/python/smg_client/types/_generated.py
	@rm -f clients/python/smg_client/types/_generated.py.bak

generate-java-types: generate-openapi ## Generate Java types from OpenAPI spec
	@echo "Generating Java types..."
	@rm -rf clients/java/src
	@npx --yes @openapitools/openapi-generator-cli@$(OPENAPI_GENERATOR_CLI_VERSION) generate \
		-i clients/openapi/smg-openapi.yaml \
		-g java \
		-o clients/java \
		--model-package com.lightseek.smg.types \
		--api-package com.lightseek.smg.api \
		--global-property models,supportingFiles,modelDocs=false,modelTests=false \
		--additional-properties serializationLibrary=jackson,dateLibrary=java8,openApiNullable=false,useJakartaEe=true,hideGenerationTimestamp=true,library=native
	@echo "Post-processing generated Java files..."
	@./scripts/fix_java_codegen.sh clients/java/src

generate-clients: generate-python-types generate-java-types ## Generate all client SDK types
	@echo "All client types generated!"

# Combined shortcuts
dev: python-dev ## Quick development setup (build Python bindings in dev mode)

install: python-install ## Build and install everything

# Release management
VERSION_FILES := model_gateway/Cargo.toml \
                 bindings/golang/Cargo.toml \
                 bindings/python/Cargo.toml \
                 bindings/python/pyproject.toml \
                 bindings/python/src/smg/version.py

show-version: ## Show current version across all files
	@echo "Current versions:"
	@echo "  model_gateway/Cargo.toml:   $$(grep -m1 '^version = ' model_gateway/Cargo.toml | sed 's/version = "\(.*\)"/\1/')"
	@echo "  bindings/golang/Cargo.toml: $$(grep -m1 '^version = ' bindings/golang/Cargo.toml | sed 's/version = "\(.*\)"/\1/')"
	@echo "  bindings/python/Cargo.toml: $$(grep -m1 '^version = ' bindings/python/Cargo.toml | sed 's/version = "\(.*\)"/\1/')"
	@echo "  bindings/python/pyproject.toml: $$(grep -m1 '^version = ' bindings/python/pyproject.toml | sed 's/version = "\(.*\)"/\1/')"
	@echo "  bindings/python/.../version.py: $$(grep '__version__' bindings/python/src/smg/version.py | sed 's/__version__ = "\(.*\)"/\1/')"

bump-version: ## Bump version across all files (usage: make bump-version VERSION=0.3.3)
	@if [ -z "$(VERSION)" ]; then \
		echo "Usage: make bump-version VERSION=<new-version>"; \
		echo "Example: make bump-version VERSION=0.3.3"; \
		echo ""; \
		echo "Current version:"; \
		grep -m1 '^version = ' model_gateway/Cargo.toml | sed 's/version = "\(.*\)"/  \1/'; \
		exit 1; \
	fi
	@echo "Bumping version to $(VERSION)..."
	@# Update model_gateway Cargo.toml
	@sed -i.bak 's/^version = ".*"/version = "$(VERSION)"/' model_gateway/Cargo.toml && rm -f model_gateway/Cargo.toml.bak
	@# Update golang binding Cargo.toml
	@sed -i.bak 's/^version = ".*"/version = "$(VERSION)"/' bindings/golang/Cargo.toml && rm -f bindings/golang/Cargo.toml.bak
	@# Update python binding Cargo.toml
	@sed -i.bak 's/^version = ".*"/version = "$(VERSION)"/' bindings/python/Cargo.toml && rm -f bindings/python/Cargo.toml.bak
	@# Update pyproject.toml
	@sed -i.bak 's/^version = ".*"/version = "$(VERSION)"/' bindings/python/pyproject.toml && rm -f bindings/python/pyproject.toml.bak
	@# Update version.py
	@sed -i.bak 's/__version__ = ".*"/__version__ = "$(VERSION)"/' bindings/python/src/smg/version.py && rm -f bindings/python/src/smg/version.py.bak
	@echo "Version updated to $(VERSION) in all files:"
	@echo "  - model_gateway/Cargo.toml"
	@echo "  - bindings/golang/Cargo.toml"
	@echo "  - bindings/python/Cargo.toml"
	@echo "  - bindings/python/pyproject.toml"
	@echo "  - bindings/python/src/smg/version.py"
	@echo ""
	@echo "Verify with: make show-version"

check-versions: ## Check workspace crate versions against latest tag (usage: make check-versions [TAG=v1.0.0] [VERSION_OVERRIDE=1.3.1])
	@VERSION_OVERRIDE="$(VERSION_OVERRIDE)" python3 scripts/check_release_versions.py $(TAG)

# Release process (scripts/release/README.md): release branches cut from main, moved by pull requests
RELEASE_TOOL := scripts/release/release.sh

release-plan: ## Before a cut: the bump the next cut writes against TAG (default: the highest final release tag), quietly; nothing is changed (usage: make release-plan [TAG=v1.13.0])
	@python3 scripts/check_release_versions.py --plan $(TAG)

release-cut: ## Cut release/<x.y> from FROM (default origin/main) and bump it to VERSION (usage: make release-cut VERSION=1.14.0-rc.1 [FROM=<sha>] [PUSH=1])
	@PUSH="$(PUSH)" $(RELEASE_TOOL) cut "$(VERSION)" $(FROM)

release-bump: ## Bump an existing release branch to VERSION through a pull request (usage: make release-bump VERSION=1.14.0-rc.2 [PUSH=0])
	@PUSH="$(PUSH)" $(RELEASE_TOOL) bump "$(VERSION)"

release-pick: ## Cherry-pick SHA (comma-separated) from main onto release/RELEASE through a pull request (usage: make release-pick RELEASE=1.14 SHA=<sha>[,<sha>] [PUSH=0])
	@PUSH="$(PUSH)" $(RELEASE_TOOL) pick "$(RELEASE)" "$(SHA)"

release-pick-pr: ## After a resolved pick conflict: push the pick branch and open its pull request (usage: make release-pick-pr RELEASE=1.14)
	@PUSH="$(PUSH)" $(RELEASE_TOOL) pick-pr "$(RELEASE)"

release-status: ## What is on main and not on release/RELEASE, its own commits, tags and open pull requests (usage: make release-status RELEASE=1.14)
	@$(RELEASE_TOOL) status "$(RELEASE)"

release-tag: ## Check the versions at release/<x.y>'s head and tag it vVERSION (usage: make release-tag VERSION=1.14.0-rc.1 [PUSH=1])
	@PUSH="$(PUSH)" $(RELEASE_TOOL) tag "$(VERSION)"

release-notes: ## The release notes skeleton for VERSION on stdout (usage: make release-notes VERSION=1.14.0 [SINCE=v1.13.0] > notes.md)
	@$(RELEASE_TOOL) notes "$(VERSION)" $(SINCE)

release-bump-main: ## After a final: a pull request that moves main's version files to vVERSION's (usage: make release-bump-main VERSION=1.14.0 [PUSH=0])
	@PUSH="$(PUSH)" $(RELEASE_TOOL) bump-main "$(VERSION)"

release-publish-plan: ## The publishing steps for vVERSION: the workflow dispatches to run against the tag and the artifacts they produce; prints, runs nothing (usage: make release-publish-plan VERSION=1.14.0)
	@$(RELEASE_TOOL) publish-plan "$(VERSION)"

release-check: ## The release gate for vVERSION: what the repository can read (versions at the branch head, its CI, the tag, main's drift) and the lines the release owner fills (the four model-provider verifiers) (usage: make release-check VERSION=1.14.0-rc.1)
	@$(RELEASE_TOOL) check "$(VERSION)"
