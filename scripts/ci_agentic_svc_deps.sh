#!/bin/bash
# CI helper for shared services (Oracle DB, Postgres DB, Brave Search MCP).
# Usage:
#   bash ci_agentic_svc_deps.sh check [--oracle <host>] [--postgres <host>] [--brave <host>]
#   bash ci_agentic_svc_deps.sh setup-oracle-client
#   bash ci_agentic_svc_deps.sh create-oracle-user <host>
#   bash ci_agentic_svc_deps.sh cleanup-oracle-user <host>
#   bash ci_agentic_svc_deps.sh create-oracle-flyway-user <host>
#   bash ci_agentic_svc_deps.sh cleanup-oracle-flyway-user <host>
#   bash ci_agentic_svc_deps.sh create-postgres-db <host>
#   bash ci_agentic_svc_deps.sh cleanup-postgres-db <host>

set -uo pipefail

# Retries with a short backoff so a shared service that's mid-restart
# (pod rollout, transient network blip) doesn't fail CI outright — but
# still hard-fails (non-zero exit) once attempts are exhausted, so a
# genuinely-down service still fails the job instead of being silently
# skipped downstream.
check_port() {
    local name="$1" host="$2" port="$3"
    local attempts="${CHECK_PORT_RETRIES:-6}" delay="${CHECK_PORT_RETRY_DELAY:-5}"
    local i
    for ((i = 1; i <= attempts; i++)); do
        echo -n "Checking $name on $host:$port (attempt $i/$attempts)... "
        if python3 -c "import socket; s=socket.create_connection(('$host', $port), 5); s.close()" 2>/dev/null; then
            echo "ok"
            return 0
        fi
        echo "not ready"
        if [ "$i" -lt "$attempts" ]; then
            sleep "$delay"
        fi
    done
    echo "FAILED: $name on $host:$port not reachable after $attempts attempts"
    return 1
}

ci_oracle_username() {
    local prefix="$1"
    local random="${CI_ORACLE_NAME_RANDOM:-$(openssl rand -hex 3)}"
    random=$(printf "%s" "$random" | tr '[:lower:]' '[:upper:]' | tr -cd 'A-Z0-9')

    local tag
    if [ -n "${GITHUB_RUN_ID:-}" ]; then
        local run_tail="${GITHUB_RUN_ID: -10}"
        local attempt="${GITHUB_RUN_ATTEMPT:-0}"
        tag="${run_tail}_${attempt}_${random}"
    else
        # Fallback for local/manual runs: keep the old hostname signal, plus entropy.
        local raw_name
        raw_name=$(echo "${HOSTNAME:-runner}" | rev | cut -d'-' -f1,2 | rev | tr '[:lower:]-' '[:upper:]_')
        tag="${raw_name}_${random}"
    fi

    tag=$(printf "%s" "$tag" | tr -cd 'A-Z0-9_')
    local max_tag_len=$((30 - ${#prefix} - 1))
    printf "%s_%s\n" "$prefix" "${tag:0:$max_tag_len}"
}

# Unique-per-run Postgres database name, so concurrent workflow runs sharing
# the `postgres-db` service don't race on first-time `CREATE TABLE`/migration
# DDL against the same database (see PR #1935 review discussion).
ci_postgres_dbname() {
    local random="${CI_POSTGRES_NAME_RANDOM:-$(openssl rand -hex 3)}"
    random=$(printf "%s" "$random" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9')

    local tag
    if [ -n "${GITHUB_RUN_ID:-}" ]; then
        local run_tail="${GITHUB_RUN_ID: -10}"
        local attempt="${GITHUB_RUN_ATTEMPT:-0}"
        tag="${run_tail}_${attempt}_${random}"
    else
        # Fallback for local/manual runs: keep the old hostname signal, plus entropy.
        local raw_name
        raw_name=$(echo "${HOSTNAME:-runner}" | rev | cut -d'-' -f1,2 | rev | tr '[:upper:]-' '[:lower:]_')
        tag="${raw_name}_${random}"
    fi

    tag=$(printf "%s" "$tag" | tr -cd 'a-z0-9_')
    printf "smg_ci_test_%s\n" "${tag:0:40}"
}

append_github_env() {
    if [ -n "${GITHUB_ENV:-}" ]; then
        echo "$1" >> "$GITHUB_ENV"
    fi
}

cmd_check() {
    local failed=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --oracle)
                if [[ -n "${2:-}" && "$2" != --* ]]; then
                    check_port "Oracle DB" "$2" 1521 || failed=1; shift 2
                else
                    check_port "Oracle DB" "oracle-db" 1521 || failed=1; shift
                fi ;;
            --postgres)
                if [[ -n "${2:-}" && "$2" != --* ]]; then
                    check_port "Postgres DB" "$2" 5432 || failed=1; shift 2
                else
                    check_port "Postgres DB" "postgres-db" 5432 || failed=1; shift
                fi ;;
            --brave)
                if [[ -n "${2:-}" && "$2" != --* ]]; then
                    check_port "Brave Search MCP" "$2" 8080 || failed=1; shift 2
                else
                    check_port "Brave Search MCP" "brave-search" 8080 || failed=1; shift
                fi ;;
            *)  echo "Unknown service: $1"; exit 1 ;;
        esac
    done
    return $failed
}

cmd_create_postgres_db() {
    set -e
    local postgres_host="${1:-postgres-db}"
    local admin_url="postgresql://postgres:postgres@${postgres_host}:5432/postgres"

    bash "$(dirname "${BASH_SOURCE[0]}")/ci_apt_mirror.sh"
    sudo apt-get update -qq
    sudo apt-get install -y -qq postgresql-client

    TEST_DB="$(ci_postgres_dbname)"
    echo "Creating Postgres test database: $TEST_DB"

    psql "$admin_url" -v ON_ERROR_STOP=1 -c "CREATE DATABASE \"$TEST_DB\";"

    append_github_env "CI_POSTGRES_TEST_DB=$TEST_DB"
    append_github_env "DATA_CONNECTOR_TEST_POSTGRES_URL=postgresql://postgres:postgres@${postgres_host}:5432/${TEST_DB}"
}

cmd_cleanup_postgres_db() {
    local postgres_host="${1:-postgres-db}"
    local admin_url="postgresql://postgres:postgres@${postgres_host}:5432/postgres"

    if [ -z "${CI_POSTGRES_TEST_DB:-}" ]; then
        echo "No Postgres test database to clean up"
        return 0
    fi

    echo "Dropping Postgres test database: $CI_POSTGRES_TEST_DB"
    psql "$admin_url" -v ON_ERROR_STOP=1 -c "DROP DATABASE IF EXISTS \"$CI_POSTGRES_TEST_DB\" WITH (FORCE);" \
        || echo "Warning: failed to drop Postgres test database $CI_POSTGRES_TEST_DB"
}

cmd_setup_oracle_client() {
    set -e
    bash "$(dirname "${BASH_SOURCE[0]}")/ci_apt_mirror.sh"
    sudo apt-get update
    sudo apt-get install -y unzip wget
    sudo apt-get install -y libaio1t64 || sudo apt-get install -y libaio1

    LIBAIO_PATH=$(find /usr/lib -name "libaio.so*" -type f 2>/dev/null | head -1)
    INSTANT_CLIENT_DIR="$HOME/instant-client"
    INSTANT_CLIENT_ZIP="instantclient-basic-linux.x64-23.26.1.0.0.zip"

    if [ ! -d "$INSTANT_CLIENT_DIR/instantclient_23_26" ]; then
        echo "Downloading Oracle Instant Client..."
        mkdir -p "$INSTANT_CLIENT_DIR"
        (cd "$INSTANT_CLIENT_DIR" &&
            wget "https://download.oracle.com/otn_software/linux/instantclient/2326100/$INSTANT_CLIENT_ZIP" &&
            unzip "$INSTANT_CLIENT_ZIP" &&
            rm "$INSTANT_CLIENT_ZIP")
    else
        echo "Oracle Instant Client already exists, skipping download"
    fi

    if [ -n "$LIBAIO_PATH" ]; then
        cp "$LIBAIO_PATH" "$INSTANT_CLIENT_DIR/instantclient_23_26/"
        ln -sf "$INSTANT_CLIENT_DIR/instantclient_23_26/$(basename "$LIBAIO_PATH")" \
            "$INSTANT_CLIENT_DIR/instantclient_23_26/libaio.so.1"
    fi

    echo "LD_LIBRARY_PATH=$INSTANT_CLIENT_DIR/instantclient_23_26:${LD_LIBRARY_PATH:-}" >> "$GITHUB_ENV"
}

cmd_create_oracle_user() {
    set -e
    local oracle_host="${1:-oracle-db}"
    local oracle_dsn="${oracle_host}:1521/FREEPDB1"

    pip install oracledb

    TEST_USER="$(ci_oracle_username TEST)"
    # Prefix with 'P' so the password always starts with a letter (Oracle requirement)
    TEST_PASS="P$(openssl rand -hex 8)"
    echo "Creating Oracle test user: $TEST_USER"

    export ORA_TEST_USER="$TEST_USER"
    export ORA_TEST_PASS="$TEST_PASS"
    export ORA_DSN="$oracle_dsn"
    append_github_env "ATP_USER=$TEST_USER"
    append_github_env "ATP_PASSWORD=$TEST_PASS"
    append_github_env "ATP_DSN=$oracle_dsn"
    append_github_env "DB_AUTO_MIGRATE=true"

    python3 << 'PYEOF'
import os, oracledb
user = os.environ["ORA_TEST_USER"]
pwd = os.environ["ORA_TEST_PASS"]
dsn = os.environ["ORA_DSN"]
conn = oracledb.connect(user="system", password="oracle", dsn=dsn)
cur = conn.cursor()
cur.execute(f'CREATE USER {user} IDENTIFIED BY "{pwd}" QUOTA UNLIMITED ON USERS')
cur.execute(f"GRANT CONNECT, RESOURCE TO {user}")
conn.commit()
conn.close()
print("Oracle test user created successfully")
PYEOF
}

cmd_cleanup_oracle_user() {
    local oracle_host="${1:-oracle-db}"
    local oracle_dsn="${oracle_host}:1521/FREEPDB1"

    if [ -z "${ATP_USER:-}" ] || [ "$ATP_USER" = "system" ]; then
        echo "No test user to clean up"
        return 0
    fi

    echo "Dropping Oracle test user: $ATP_USER"
    pip install oracledb 2>/dev/null || true

    export ORA_DROP_USER="$ATP_USER"
    export ORA_DSN="$oracle_dsn"

    python3 << 'PYEOF' || echo "Warning: cleanup script failed"
import os, oracledb
try:
    user = os.environ["ORA_DROP_USER"]
    dsn = os.environ["ORA_DSN"]
    conn = oracledb.connect(user="system", password="oracle", dsn=dsn)
    cur = conn.cursor()
    cur.execute(f"DROP USER {user} CASCADE")
    conn.commit()
    conn.close()
    print("Oracle test user dropped successfully")
except Exception as e:
    print(f"Warning: failed to drop test user: {e}")
PYEOF
}

cmd_create_oracle_flyway_user() {
    set -e
    local oracle_host="${1:-oracle-db}"
    local oracle_dsn="${oracle_host}:1521/FREEPDB1"

    pip install oracledb

    FLYWAY_USER="$(ci_oracle_username FLYWAY)"
    # Prefix with 'P' so the password always starts with a letter (Oracle requirement)
    FLYWAY_PASS="P$(openssl rand -hex 8)"
    echo "Creating Oracle Flyway test user: $FLYWAY_USER"

    export ORA_FLYWAY_USER="$FLYWAY_USER"
    export ORA_FLYWAY_PASS="$FLYWAY_PASS"
    export ORA_DSN="$oracle_dsn"
    append_github_env "ATP_FLYWAY_USER=$FLYWAY_USER"
    append_github_env "ATP_FLYWAY_PASSWORD=$FLYWAY_PASS"
    append_github_env "ATP_FLYWAY_DSN=$oracle_dsn"

    # Locate Flyway SQL files relative to the repo root
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    export FLYWAY_SQL_DIR="$REPO_ROOT/scripts/oracle_flyway/sql"

    python3 << 'PYEOF'
import os, glob, oracledb

user = os.environ["ORA_FLYWAY_USER"]
pwd = os.environ["ORA_FLYWAY_PASS"]
dsn = os.environ["ORA_DSN"]
sql_dir = os.environ["FLYWAY_SQL_DIR"]

# Create user with extra privileges needed for V2 sweep procedures/scheduler jobs
conn = oracledb.connect(user="system", password="oracle", dsn=dsn)
cur = conn.cursor()
cur.execute(f'CREATE USER {user} IDENTIFIED BY "{pwd}" QUOTA UNLIMITED ON USERS')
cur.execute(f"GRANT CONNECT, RESOURCE TO {user}")
conn.commit()
conn.close()
print(f"Oracle Flyway user {user} created successfully")

# Run Flyway SQL files against the new user
flyway_conn = oracledb.connect(user=user, password=pwd, dsn=dsn)
flyway_cur = flyway_conn.cursor()

sql_files = sorted(glob.glob(os.path.join(sql_dir, "V*.sql")))
for sql_file in sql_files:
    print(f"Executing {os.path.basename(sql_file)}...")
    with open(sql_file) as f:
        content = f.read()

    # Split on semicolons and execute each statement
    for stmt in content.split(";"):
        # Strip comments and whitespace
        lines = [l for l in stmt.split("\n") if not l.strip().startswith("--")]
        cleaned = "\n".join(lines).strip()
        if cleaned:
            flyway_cur.execute(cleaned)

flyway_conn.commit()
flyway_conn.close()
print("Flyway SQL files executed successfully")
PYEOF
}

cmd_cleanup_oracle_flyway_user() {
    local oracle_host="${1:-oracle-db}"
    local oracle_dsn="${oracle_host}:1521/FREEPDB1"

    if [ -z "${ATP_FLYWAY_USER:-}" ] || [ "$ATP_FLYWAY_USER" = "system" ]; then
        echo "No Flyway test user to clean up"
        return 0
    fi

    echo "Dropping Oracle Flyway test user: $ATP_FLYWAY_USER"
    pip install oracledb 2>/dev/null || true

    export ORA_DROP_USER="$ATP_FLYWAY_USER"
    export ORA_DSN="$oracle_dsn"

    python3 << 'PYEOF' || echo "Warning: cleanup script failed"
import os, oracledb
try:
    user = os.environ["ORA_DROP_USER"]
    dsn = os.environ["ORA_DSN"]
    conn = oracledb.connect(user="system", password="oracle", dsn=dsn)
    cur = conn.cursor()
    cur.execute(f"DROP USER {user} CASCADE")
    conn.commit()
    conn.close()
    print("Oracle Flyway test user dropped successfully")
except Exception as e:
    print(f"Warning: failed to drop Flyway test user: {e}")
PYEOF
}

if [ "${CI_AGENTIC_SVC_DEPS_LIB_ONLY:-0}" = "1" ]; then
    return 0 2>/dev/null || exit 0
fi

COMMAND="${1:?Usage: ci_agentic_svc_deps.sh <command> [args...]}"
shift

case "$COMMAND" in
    check)                cmd_check "$@" ;;
    setup-oracle-client)  cmd_setup_oracle_client ;;
    create-oracle-user)   cmd_create_oracle_user "$@" ;;
    cleanup-oracle-user)  cmd_cleanup_oracle_user "$@" ;;
    create-oracle-flyway-user)   cmd_create_oracle_flyway_user "$@" ;;
    cleanup-oracle-flyway-user)  cmd_cleanup_oracle_flyway_user "$@" ;;
    create-postgres-db)   cmd_create_postgres_db "$@" ;;
    cleanup-postgres-db)  cmd_cleanup_postgres_db "$@" ;;
    *)                    echo "Unknown command: $COMMAND"; exit 1 ;;
esac
