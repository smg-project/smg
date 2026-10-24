"""Require the successful current-main build; the parent run is reported separately."""

POLICY = "successful_build_job_with_consumer_revalidation"
SUCCESS_STEPS = [
    "Set up job",
    "Run actions/checkout@v7",
    "Setup Rust",
    "Build Python wheel and Go FFI library",
    "Generate Python client types",
    "Build WASM test fixtures",
    "Upload wheel artifact",
    "Upload Go FFI library artifact",
    "Upload Python client types",
    "Upload WASM test fixtures",
    "Show sccache stats",
    "Set up Python",
    "Test wheel install",
    "Post Set up Python",
    "Post Run actions/checkout@v7",
    "Complete job",
]
ARTIFACTS = {"smg-wheel", "python-client-types", "wasm-test-fixtures", "go-ffi-library"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(run, job, inputs):
    require(inputs["producer_policy"] == POLICY, "unknown successful-build policy")
    require(
        run["id"] == inputs["run_id"]
        and run["head_sha"] == inputs["producer_sha"]
        and run["run_attempt"] == inputs["run_attempt"]
        and run["event"] == "push"
        and run["path"] == ".github/workflows/pr-test-rust.yml"
        and run["head_branch"] == "main"
        and run["status"] in ("queued", "in_progress", "completed"),
        "producer run identity",
    )
    require(
        job["id"] == inputs["build_job_id"]
        and job["run_id"] == inputs["run_id"]
        and job["run_attempt"] == inputs["run_attempt"]
        and job["head_sha"] == inputs["producer_sha"]
        and job["name"] == "build-wheel"
        and job["status"] == "completed"
        and job["conclusion"] == "success",
        "producer build identity or terminal state",
    )
    steps = job["steps"]
    names = [step["name"] for step in steps]
    require(
        len(names) == len(set(names)) and set(names) == {*SUCCESS_STEPS, "Post Setup Rust"},
        "producer step identity changed",
    )
    for step in steps:
        expected = "success"
        require(
            step["status"] == "completed" and step["conclusion"] == expected,
            "critical producer step not successful: " + step["name"],
        )
    require(
        len(inputs["artifacts"]) == 4
        and {a["name"] for a in inputs["artifacts"]} == ARTIFACTS
        and len({a["id"] for a in inputs["artifacts"]}) == 4,
        "four distinct standard artifacts required",
    )


def unique_artifacts(records, inputs):
    for expected in inputs["artifacts"]:
        matching = [item for item in records if item["name"] == expected["name"]]
        require(
            len(matching) == 1 and matching[0]["id"] == expected["id"],
            "producer artifact name replaced or duplicated",
        )
