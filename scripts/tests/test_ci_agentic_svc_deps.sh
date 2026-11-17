#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CI_AGENTIC_SVC_DEPS_LIB_ONLY=1 source "$REPO_ROOT/scripts/ci_agentic_svc_deps.sh"

assert_eq() {
    local expected="$1" actual="$2"
    if [ "$expected" != "$actual" ]; then
        echo "expected: $expected"
        echo "actual:   $actual"
        exit 1
    fi
}

assert_matches() {
    local value="$1" pattern="$2"
    if [[ ! "$value" =~ $pattern ]]; then
        echo "expected '$value' to match /$pattern/"
        exit 1
    fi
}

assert_max_len() {
    local value="$1" max="$2"
    if [ "${#value}" -gt "$max" ]; then
        echo "expected '$value' to be at most $max chars, got ${#value}"
        exit 1
    fi
}

test_run_scoped_oracle_usernames_are_unique_and_oracle_safe() {
    local test_user_a
    test_user_a=$(
        HOSTNAME=2-gpu-h100 \
            GITHUB_RUN_ID=27463234653 \
            GITHUB_RUN_ATTEMPT=4 \
            CI_ORACLE_NAME_RANDOM=A1B2C3 \
            ci_oracle_username TEST
    )
    assert_eq "TEST_7463234653_4_A1B2C3" "$test_user_a"

    local test_user_b
    test_user_b=$(
        HOSTNAME=2-gpu-h100 \
            GITHUB_RUN_ID=27463234653 \
            GITHUB_RUN_ATTEMPT=4 \
            CI_ORACLE_NAME_RANDOM=D4E5F6 \
            ci_oracle_username TEST
    )
    assert_eq "TEST_7463234653_4_D4E5F6" "$test_user_b"

    local flyway_user
    flyway_user=$(
        HOSTNAME=2-gpu-h100 \
            GITHUB_RUN_ID=27463234653 \
            GITHUB_RUN_ATTEMPT=4 \
            CI_ORACLE_NAME_RANDOM=A1B2C3 \
            ci_oracle_username FLYWAY
    )
    assert_eq "FLYWAY_7463234653_4_A1B2C3" "$flyway_user"
    assert_max_len "$flyway_user" 30
}

test_hostname_fallback_keeps_entropy_and_stays_under_oracle_limit() {
    local username
    username=$(
        HOSTNAME=arc-runner-gpu-h100-jfvzm-m2lm8 \
            GITHUB_RUN_ID= \
            GITHUB_RUN_ATTEMPT= \
            CI_ORACLE_NAME_RANDOM=ABCDEF \
            ci_oracle_username TEST
    )

    assert_matches "$username" '^TEST_[A-Z0-9_]+_ABCDEF$'
    assert_max_len "$username" 30
}

test_postgres_apt_case() (
    local scenario="$1" expected_status="$2" expected_updates="$3"
    local expected_installs="$4" expected_mirrors="$5" expected_reprobes="$6"
    local updates=0 installs=0 mirrors=0 reprobes=0 ddl=0 status=0
    local temporary
    temporary="$(mktemp -d "${TMPDIR:-/tmp}/agentic-apt.XXXXXX")"
    trap 'rm -rf "$temporary"' EXIT
    export GITHUB_RUN_ID=43966383110 GITHUB_RUN_ATTEMPT=1 CI_POSTGRES_NAME_RANDOM=AABBCC
    export GITHUB_ENV="$temporary/github-env"
    : > "$GITHUB_ENV"
    unset TEST_DB CI_POSTGRES_TEST_DB DATA_CONNECTOR_TEST_POSTGRES_URL

    # Intercept the real helper's commands, not a reimplementation of its
    # retry logic. These functions are scoped to this test's subshell.
    bash() {
        assert_eq "$REPO_ROOT/scripts/ci_apt_mirror.sh" "$1"
        mirrors=$((mirrors + 1))
        if [ "$#" -eq 2 ]; then
            assert_eq --reprobe "$2"
            reprobes=$((reprobes + 1))
            [ "$scenario" != reprobe_fail ] || return 73
        else
            assert_eq 1 "$#"
            [ "$scenario" != mirror_fail ] || return 71
        fi
        return 0
    }
    sudo() {
        assert_eq apt-get "$1"
        shift
        case "$1" in
            update)
                assert_eq 'update -qq' "$*"
                updates=$((updates + 1))
                [ "$scenario" != initial_update_fail ] || return 72
                if [ "$scenario" = retry_update_fail ] && [ "$updates" -eq 2 ]; then
                    return 74
                fi
                ;;
            install)
                assert_eq 'install -y -qq postgresql-client' "$*"
                installs=$((installs + 1))
                if [ "$scenario" = second_install_fail ]; then
                    return 100
                fi
                if [ "$scenario" != success ] && [ "$installs" -eq 1 ]; then
                    return 100
                fi
                ;;
            *) echo "unexpected apt operation: $*" >&2; exit 1 ;;
        esac
        return 0
    }
    psql() {
        ddl=$((ddl + 1))
        assert_eq 5 "$#"
        assert_eq 'postgresql://postgres:postgres@postgres-db:5432/postgres' "$1"
        assert_eq -v "$2"
        assert_eq ON_ERROR_STOP=1 "$3"
        assert_eq -c "$4"
        assert_eq 'CREATE DATABASE "smg_ci_test_3966383110_1_aabbcc";' "$5"
    }

    # A function invoked in an if-condition cannot rely on set -e. Every
    # failed setup command must still propagate and prevent DDL/env export.
    if cmd_create_postgres_db postgres-db; then
        status=0
    else
        status=$?
    fi
    assert_eq "$expected_status" "$status"
    assert_eq "$expected_updates" "$updates"
    assert_eq "$expected_installs" "$installs"
    assert_eq "$expected_mirrors" "$mirrors"
    assert_eq "$expected_reprobes" "$reprobes"
    if [ "$status" -eq 0 ]; then
        assert_eq 1 "$ddl"
        assert_eq $'CI_POSTGRES_TEST_DB=smg_ci_test_3966383110_1_aabbcc\nDATA_CONNECTOR_TEST_POSTGRES_URL=postgresql://postgres:postgres@postgres-db:5432/smg_ci_test_3966383110_1_aabbcc' "$(cat "$GITHUB_ENV")"
    else
        assert_eq 0 "$ddl"
        assert_eq '' "$(cat "$GITHUB_ENV")"
    fi
    echo "PASS postgres_apt_$scenario"
)

test_mirror_choice_case() (
    local scenario="$1" temporary current alternate expected calls
    temporary="$(mktemp -d "${TMPDIR:-/tmp}/agentic-mirror.XXXXXX")"
    trap 'rm -rf "$temporary"' EXIT
    current=http://eu-frankfurt-1.clouds.archive.ubuntu.com/ubuntu
    alternate=http://azure.archive.ubuntu.com/ubuntu
    mkdir -p "$temporary/apt/apt.conf.d" "$temporary/apt/sources.list.d" "$temporary/bin"
    printf 'ID=ubuntu\nVERSION_CODENAME=noble\n' > "$temporary/os-release"
    printf 'deb %s/ noble main\ndeb %s/ noble-security main\n' "$current" "$current" > "$temporary/apt/sources.list"
    printf '// Written by scripts/ci_apt_mirror.sh (mirror=%s).\n' "$current" > "$temporary/apt/apt.conf.d/99-ci-mirror"
    export CI_APT_ROOT="$temporary/apt" CI_OS_RELEASE="$temporary/os-release"
    export CI_APT_MIRRORS="$current $alternate" CURL_LOG="$temporary/curl-calls"
    export MIRROR_TEST_CASE="$scenario"
    : > "$CURL_LOG"
    cat > "$temporary/bin/curl" <<'EOF'
#!/bin/bash
set -euo pipefail
url="${!#}"
printf '%s\n' "$url" >> "$CURL_LOG"
if [ "$MIRROR_TEST_CASE" = fallback_current ] && [[ "$url" = http://azure.archive.ubuntu.com/* ]]; then
    exit 22
fi
exit 0
EOF
    chmod +x "$temporary/bin/curl"
    export PATH="$temporary/bin:$PATH"
    if [ "$scenario" = cached_default ]; then
        bash "$REPO_ROOT/scripts/ci_apt_mirror.sh"
        expected="$current"
        assert_eq '' "$(cat "$CURL_LOG")"
    else
        bash "$REPO_ROOT/scripts/ci_apt_mirror.sh" --reprobe
        if [ "$scenario" = alternate ]; then
            expected="$alternate"
            calls="$(cat "$CURL_LOG")"
            assert_eq "$alternate/dists/noble/InRelease"$'\n'"$alternate/dists/noble-security/InRelease" "$calls"
        else
            assert_eq fallback_current "$scenario"
            expected="$current"
            calls="$(cat "$CURL_LOG")"
            assert_eq "$alternate/dists/noble/InRelease"$'\n'"$current/dists/noble/InRelease"$'\n'"$current/dists/noble-security/InRelease" "$calls"
        fi
    fi
    grep -Fq "mirror=$expected" "$temporary/apt/apt.conf.d/99-ci-mirror"
    assert_eq "deb $expected/ noble main"$'\n'"deb $expected/ noble-security main" "$(cat "$temporary/apt/sources.list")"
    echo "PASS apt_mirror_$scenario"
)

test_run_scoped_oracle_usernames_are_unique_and_oracle_safe
echo 'PASS original_oracle_run_scoped_names'
test_hostname_fallback_keeps_entropy_and_stays_under_oracle_limit
echo 'PASS original_oracle_hostname_names'
test_postgres_apt_case success 0 1 1 1 0
test_postgres_apt_case install_then_success 0 2 2 2 1
test_postgres_apt_case second_install_fail 100 2 2 2 1
test_postgres_apt_case mirror_fail 71 0 0 1 0
test_postgres_apt_case initial_update_fail 72 1 0 1 0
test_postgres_apt_case reprobe_fail 73 1 1 2 1
test_postgres_apt_case retry_update_fail 74 2 1 2 1
test_mirror_choice_case cached_default
test_mirror_choice_case alternate
test_mirror_choice_case fallback_current

echo "ci_agentic_svc_deps tests passed"
