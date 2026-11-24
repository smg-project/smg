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
    local ddl=0 status=0 temporary
    temporary="$(mktemp -d "${TMPDIR:-/tmp}/agentic-apt.XXXXXX")"
    trap 'rm -rf "$temporary"' EXIT
    export GITHUB_RUN_ID=43966383110 GITHUB_RUN_ATTEMPT=1 CI_POSTGRES_NAME_RANDOM=AABBCC
    export GITHUB_ENV="$temporary/github-env"
    : > "$GITHUB_ENV"
    export FORBIDDEN_LOG="$temporary/forbidden"
    : > "$FORBIDDEN_LOG"
    unset TEST_DB CI_POSTGRES_TEST_DB DATA_CONNECTOR_TEST_POSTGRES_URL
    mkdir -p "$temporary/scripts" "$temporary/bin" "$temporary/apt/sources.list.d"
    # Exercise the actual three-attempt central install/update/retry code.
    # Only mirror/network and apt executables are private mocks. Persistent
    # files count across the real subprocess boundaries, not exported funcs.
    cp "$REPO_ROOT/scripts/ci_agentic_svc_deps.sh" "$temporary/scripts/"
    cp "$REPO_ROOT/scripts/ci_apt_install.sh" "$REPO_ROOT/scripts/ci_apt_update.sh" "$REPO_ROOT/scripts/ci_retry.sh" "$temporary/scripts/"
    CI_AGENTIC_SVC_DEPS_LIB_ONLY=1 source "$temporary/scripts/ci_agentic_svc_deps.sh"
    printf 'deb http://archive.ubuntu.com/ubuntu/ noble main\n' > "$temporary/apt/sources.list"
    export CI_APT_ROOT="$temporary/apt" CI_APT_INSTALL_RETRY_DELAY=0 CI_APT_UPDATE_RETRY_DELAY=0
    export PG_APT_SCENARIO="$scenario" PG_APT_COUNTS="$temporary"
    : > "$temporary/updates"; : > "$temporary/installs"; : > "$temporary/mirrors"; : > "$temporary/reprobes"
    cat > "$temporary/scripts/ci_apt_mirror.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
printf 'call\n' >> "$PG_APT_COUNTS/mirrors"
if [ "$#" -eq 1 ]; then
    [ "$1" = --reprobe ] || exit 97
    printf 'call\n' >> "$PG_APT_COUNTS/reprobes"
    [ "$PG_APT_SCENARIO" != reprobe_fail ] || exit 73
else
    [ "$#" -eq 0 ] || exit 97
fi
[ "$PG_APT_SCENARIO" != mirror_fail ] || exit 71
EOF
    cat > "$temporary/bin/sudo" <<'EOF'
#!/bin/bash
set -euo pipefail
[ "$1" = apt-get ] || { echo 'forbidden sudo command' >> "$FORBIDDEN_LOG"; exit 97; }
shift
exec apt-get "$@"
EOF
    cat > "$temporary/bin/apt-get" <<'EOF'
#!/bin/bash
set -euo pipefail
case "$1" in
    update)
        printf 'call\n' >> "$PG_APT_COUNTS/updates"
        [ "$PG_APT_SCENARIO" != initial_update_fail ] || exit 72
        if [ "$PG_APT_SCENARIO" = retry_update_fail ] && [ "$(wc -l < "$PG_APT_COUNTS/updates")" -ge 2 ]; then exit 74; fi
        ;;
    install)
        [ "$*" = 'install -y -qq postgresql-client' ] || exit 97
        printf 'call\n' >> "$PG_APT_COUNTS/installs"
        [ "$PG_APT_SCENARIO" != second_install_fail ] || exit 100
        if [ "$PG_APT_SCENARIO" != success ] && [ "$PG_APT_SCENARIO" != psql_fail ] \
            && [ "$(wc -l < "$PG_APT_COUNTS/installs")" -eq 1 ]; then exit 100; fi
        ;;
    *) echo 'forbidden apt operation' >> "$FORBIDDEN_LOG"; exit 97 ;;
esac
EOF
    cat > "$temporary/bin/sleep" <<'EOF'
#!/bin/bash
[ "$1" = 0 ] || exit 97
EOF
    chmod +x "$temporary/bin/"* "$temporary/scripts/ci_apt_mirror.sh"
    export PATH="$temporary/bin:$PATH"
    psql() {
        ddl=$((ddl + 1))
        assert_eq 5 "$#"
        assert_eq 'postgresql://postgres:postgres@postgres-db:5432/postgres' "$1"
        assert_eq -v "$2"
        assert_eq ON_ERROR_STOP=1 "$3"
        assert_eq -c "$4"
        assert_eq 'CREATE DATABASE "smg_ci_test_3966383110_1_aabbcc";' "$5"
        [ "$PG_APT_SCENARIO" != psql_fail ] || return 2
    }
    # A function called in an if-condition must propagate failure explicitly.
    if cmd_create_postgres_db postgres-db; then status=0; else status=$?; fi
    assert_eq "$expected_status" "$status"
    assert_eq "$expected_updates" "$(wc -l < "$temporary/updates")"
    assert_eq "$expected_installs" "$(wc -l < "$temporary/installs")"
    assert_eq "$expected_mirrors" "$(wc -l < "$temporary/mirrors")"
    assert_eq "$expected_reprobes" "$(wc -l < "$temporary/reprobes")"
    if [ "$status" -eq 0 ]; then
        assert_eq 1 "$ddl"
        assert_eq $'CI_POSTGRES_TEST_DB=smg_ci_test_3966383110_1_aabbcc\nDATA_CONNECTOR_TEST_POSTGRES_URL=postgresql://postgres:postgres@postgres-db:5432/smg_ci_test_3966383110_1_aabbcc' "$(cat "$GITHUB_ENV")"
    else
        if [ "$scenario" = psql_fail ]; then assert_eq 1 "$ddl"; else assert_eq 0 "$ddl"; fi
        assert_eq '' "$(cat "$GITHUB_ENV")"
    fi
    assert_eq '' "$(cat "$FORBIDDEN_LOG")"
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


# Exercise real mirror rewriting with private apt roots and a curl stub only.
test_custom_mirror_reprobe_case() (
    local scenario="$1" temporary current alternate expected
    temporary="$(mktemp -d "${TMPDIR:-/tmp}/agentic-custom-mirror.XXXXXX")"
    trap 'rm -rf "$temporary"' EXIT
    current='https://mirror.example/a.[x]+(repo)'
    alternate='https://backup.example/repo&extras'
    if [ "$scenario" = main_first ]; then
        current=http://archive.ubuntu.com/ubuntu
    fi
    mkdir -p "$temporary/apt/apt.conf.d" "$temporary/apt/sources.list.d" "$temporary/bin"
    printf 'ID=ubuntu\nVERSION_CODENAME=noble\n' > "$temporary/os-release"
    printf 'deb %s/ noble main\ndeb %s/ noble-security main\n' "$current" "$current" > "$temporary/apt/sources.list"
    printf 'Types: deb\nURIs: %s/\nSuites: noble noble-security\nComponents: main\n' "$current" > "$temporary/apt/sources.list.d/ubuntu.sources"
    printf 'deb http://ports.ubuntu.com/ubuntu-ports noble main\ndeb https://mirror.example/aZxrepo noble main\ndeb %s-other noble main\n' "$current" > "$temporary/apt/sources.list.d/unrelated.list"
    cp "$temporary/apt/sources.list.d/unrelated.list" "$temporary/unrelated-before"
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
[ "$MIRROR_TEST_CASE" != custom_none ]
EOF
    chmod +x "$temporary/bin/curl"
    export PATH="$temporary/bin:$PATH"
    if [ "$scenario" = custom_alternate ]; then
        # Real first selection writes custom A; the next call must discover
        # that exact origin in both legacy and deb822 sources and move to B.
        rm "$temporary/apt/apt.conf.d/99-ci-mirror"
        printf 'deb http://archive.ubuntu.com/ubuntu/ noble main\ndeb http://security.ubuntu.com/ubuntu/ noble-security main\n' > "$temporary/apt/sources.list"
        printf 'Types: deb\nURIs: http://archive.ubuntu.com/ubuntu/\nSuites: noble noble-security\nComponents: main\n' > "$temporary/apt/sources.list.d/ubuntu.sources"
        bash "$REPO_ROOT/scripts/ci_apt_mirror.sh" --reprobe
        assert_eq "deb $current/ noble main"$'\n'"deb $current/ noble-security main" "$(cat "$temporary/apt/sources.list")"
        : > "$CURL_LOG"
    fi
    if [ "$scenario" = custom_cached ]; then
        bash "$REPO_ROOT/scripts/ci_apt_mirror.sh"
        expected="$current"
        assert_eq '' "$(cat "$CURL_LOG")"
    elif [ "$scenario" = custom_none ]; then
        cp "$temporary/apt/sources.list" "$temporary/source-before"
        bash "$REPO_ROOT/scripts/ci_apt_mirror.sh" --reprobe
        cmp "$temporary/source-before" "$temporary/apt/sources.list"
        grep -Fq 'mirror=none' "$temporary/apt/apt.conf.d/99-ci-mirror"
        expected="$current"
    else
        bash "$REPO_ROOT/scripts/ci_apt_mirror.sh" --reprobe
        expected="$alternate"
        assert_eq "$alternate/dists/noble/InRelease"$'\n'"$alternate/dists/noble-security/InRelease" "$(cat "$CURL_LOG")"
    fi
    assert_eq "deb $expected/ noble main"$'\n'"deb $expected/ noble-security main" "$(cat "$temporary/apt/sources.list")"
    grep -Fxq "URIs: $expected/" "$temporary/apt/sources.list.d/ubuntu.sources"
    cmp "$temporary/unrelated-before" "$temporary/apt/sources.list.d/unrelated.list"
    echo "PASS apt_mirror_$scenario"
)

test_mirror_current_origin_case() (
    local scenario="$1" temporary main cloud alternate expected
    temporary="$(mktemp -d "${TMPDIR:-/tmp}/agentic-origin-mirror.XXXXXX")"
    trap 'rm -rf "$temporary"' EXIT
    main=http://archive.ubuntu.com/ubuntu
    cloud=http://eu-frankfurt-1.clouds.archive.ubuntu.com/ubuntu
    alternate=http://azure.archive.ubuntu.com/ubuntu
    mkdir -p "$temporary/apt/apt.conf.d" "$temporary/apt/sources.list.d" "$temporary/bin"
    printf 'ID=ubuntu\nVERSION_CODENAME=noble\n' > "$temporary/os-release"
    case "$scenario" in
        ordinary_https_main)
            printf 'deb https://archive.ubuntu.com/ubuntu/ noble main\ndeb https://security.ubuntu.com/ubuntu/ noble-security main\n' > "$temporary/apt/sources.list"
            expected="$alternate"
            export CI_APT_MIRRORS="$alternate $cloud $main"
            ;;
        ordinary_mixed)
            printf 'deb %s/ noble main\ndeb %s/ noble-updates main\n' "$main" "$cloud" > "$temporary/apt/sources.list"
            expected="$cloud"
            export CI_APT_MIRRORS="$alternate $cloud $main"
            ;;
        reprobe_main_fallback_mixed)
            printf 'deb %s/ noble main\ndeb %s/ noble-updates main\n' "$main" "$cloud" > "$temporary/apt/sources.list"
            expected="$main"
            export CI_APT_MIRRORS="$main $cloud"
            ;;
        *) echo 'unknown origin scenario' >&2; exit 1 ;;
    esac
    export CI_APT_ROOT="$temporary/apt" CI_OS_RELEASE="$temporary/os-release"
    export CURL_LOG="$temporary/curl-calls" MIRROR_TEST_CASE="$scenario"
    : > "$CURL_LOG"
    cat > "$temporary/bin/curl" <<'EOF'
#!/bin/bash
set -euo pipefail
url="${!#}"
printf '%s\n' "$url" >> "$CURL_LOG"
if [ "$MIRROR_TEST_CASE" = reprobe_main_fallback_mixed ] && [[ "$url" = http://eu-frankfurt-1.clouds.archive.ubuntu.com/* ]]; then
    exit 22
fi
exit 0
EOF
    chmod +x "$temporary/bin/curl"
    export PATH="$temporary/bin:$PATH"
    if [ "$scenario" = reprobe_main_fallback_mixed ]; then
        bash "$REPO_ROOT/scripts/ci_apt_mirror.sh" --reprobe
        assert_eq "deb $expected/ noble main"$'\n'"deb $expected/ noble-updates main" "$(cat "$temporary/apt/sources.list")"
    else
        bash "$REPO_ROOT/scripts/ci_apt_mirror.sh"
        assert_eq "$expected/dists/noble/InRelease"$'\n'"$expected/dists/noble-security/InRelease" "$(cat "$CURL_LOG")"
        grep -Fq "deb $expected/ noble main" "$temporary/apt/sources.list"
        # The old main must join the existing healthy cloud too.
        if [ "$scenario" = ordinary_mixed ]; then
            assert_eq "deb $expected/ noble main"$'\n'"deb $expected/ noble-updates main" "$(cat "$temporary/apt/sources.list")"
        fi
    fi
    grep -Fq "mirror=$expected" "$temporary/apt/apt.conf.d/99-ci-mirror"
    echo "PASS apt_mirror_$scenario"
)

test_run_scoped_oracle_usernames_are_unique_and_oracle_safe
echo 'PASS original_oracle_run_scoped_names'
test_hostname_fallback_keeps_entropy_and_stays_under_oracle_limit
echo 'PASS original_oracle_hostname_names'
test_postgres_apt_case success 0 1 1 1 0
test_postgres_apt_case install_then_success 0 2 2 2 1
test_postgres_apt_case second_install_fail 100 3 3 3 2
test_postgres_apt_case mirror_fail 71 0 0 3 2
test_postgres_apt_case initial_update_fail 72 6 0 3 2
test_postgres_apt_case reprobe_fail 73 1 1 3 2
test_postgres_apt_case retry_update_fail 74 5 1 3 2
test_mirror_choice_case cached_default
test_mirror_choice_case alternate
test_mirror_choice_case fallback_current

test_custom_mirror_reprobe_case custom_alternate
test_custom_mirror_reprobe_case main_first
test_custom_mirror_reprobe_case custom_cached
test_custom_mirror_reprobe_case custom_none

test_mirror_current_origin_case ordinary_https_main
test_mirror_current_origin_case ordinary_mixed
test_mirror_current_origin_case reprobe_main_fallback_mixed

# A failed CREATE must not publish a database that does not exist.
test_postgres_apt_case psql_fail 2 1 1 1 0

echo "ci_agentic_svc_deps tests passed"
