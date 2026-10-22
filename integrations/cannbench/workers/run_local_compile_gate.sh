#!/usr/bin/env bash
# Bounded, FAIL-CLOSED local compile gate.
#
# The shared execution lock must never be released while the compile container
# may still be alive. Docker CLI transport/daemon errors are NOT proof of
# absence, so release requires a CONFIRMED-absent container. On timeout the planned case count is preserved (never 0).
#
#   PYASC_CANN_GATE_TIMEOUT_SECONDS       (default 900)  bound on docker run
#   PYASC_CANN_GATE_BUILD_TIMEOUT_SECONDS (default 1800) bound on docker build
#   PYASC_CANN_GATE_CONTAINER             deterministic container name (driver)
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
image="pyasc-cannbench-local-eval:0a631f70-v2"

if [ "${PYASC_CANN_LOCK_HELD:-}" != execution ]; then
    exec python3 "${repo_root}/integrations/cannbench/ci/vm_locks.py" \
        --kind execution -- bash "$0" "$@"
fi

# This marker is emitted only after vm_locks has acquired both locks.
if [ -n "${PYASC_CANN_GATE_LOCK_MARKER:-}" ]; then
    printf 'acquired\n' > "$PYASC_CANN_GATE_LOCK_MARKER"
fi

is_positive_int() {
    case "${1:-}" in
        ''|*[!0-9]*) return 1 ;;
    esac
    [ "$1" -gt 0 ]
}

gate_timeout="${PYASC_CANN_GATE_TIMEOUT_SECONDS:-900}"
build_timeout="${PYASC_CANN_GATE_BUILD_TIMEOUT_SECONDS:-1800}"
for value in "${gate_timeout}" "${build_timeout}"; do
    if ! is_positive_int "${value}"; then
        echo "[gate] invalid timeout '${value}' (positive integer required)" >&2
        exit 2
    fi
done

container_name="${PYASC_CANN_GATE_CONTAINER:-pyasc-cannbench-gate-$$-$(date +%s)}"

planned_cases() {
    local cases_file="" previous=""
    for argument in "$@"; do
        [ "${previous}" = "--cases" ] && cases_file="${argument}"
        previous="${argument}"
    done
    if [ -n "${cases_file}" ] && [ -f "${repo_root}/${cases_file}" ]; then
        grep -c "case_id" "${repo_root}/${cases_file}" || true
    else
        echo 20
    fi
}

emit_report() {
    local status="$1" reason="$2" cases="$3" confirmed="$4"
    printf '{"operator":"%s","status":"%s","fatal_error":"%s","gate_timed_out":%s,"timeout_seconds":%s,"container":"%s","container_confirmed_stopped":%s,"cases":%s,"planned_cases":%s,"compile_passed":0,"dispatch_passed":0,"case_results":[]}\n' \
        "${gate_op}" "${status}" "${reason}" \
        "$([ "${status}" = failed ] && echo true || echo false)" \
        "${gate_timeout}" "${container_name}" "${confirmed}" \
        "${cases}" "${cases}"
}

container_state() {
    # echoes running | present | absent | unknown. Never guesses on daemon errors.
    local output rc
    output="$(docker inspect --type container --format '{{.State.Running}}' "$1" 2>&1)"
    rc=$?
    if [ "${rc}" -eq 0 ]; then
        if [ "${output}" = "true" ]; then echo running; else echo present; fi
        return
    fi
    case "${output}" in
        *"No such object"*|*"No such container"*|*"no such"*) echo absent ;;
        *) echo unknown ;;
    esac
}

daemon_available() {
    docker version --format '{{.Server.Version}}' >/dev/null 2>&1
}

remove_container() {
    docker rm -f "$1" >/dev/null 2>&1 || true
}

# 0 = confirmed absent;
# 1 = unreconciled (container may still be alive -> must retain the lock).
reconcile_stop() {
    local name="$1" budget="${2:-120}" deadline consecutive=0 state
    deadline=$(( $(date +%s) + budget ))
    while :; do
        state="$(container_state "${name}")"
        case "${state}" in
            absent)
                consecutive=$((consecutive + 1))
                [ "${consecutive}" -ge 2 ] && return 0
                ;;
            running|present)
                consecutive=0
                remove_container "${name}"
                ;;
            unknown)
                consecutive=0
                # Daemon unavailability does not prove container termination.
                # Retain the lock, including during Docker live restore.
                ;;
        esac
        [ "$(date +%s)" -ge "${deadline}" ] && return 1
        sleep 0.5
    done
}

gate_op=""
previous=""
for argument in "$@"; do
    [ "${previous}" = "--op" ] && gate_op="${argument}"
    previous="${argument}"
done
case_count="$(planned_cases "$@")"

if ! docker image inspect "${image}" >/dev/null 2>&1; then
    set +e
    timeout --signal=TERM --kill-after=30 "${build_timeout}" \
        docker build \
        --platform linux/amd64 \
        --file "${repo_root}/integrations/cannbench/Dockerfile.local-eval" \
        --tag "${image}" \
        "${repo_root}/integrations/cannbench" >&2
    build_rc=$?
    set -e
    if [ "${build_rc}" -ne 0 ]; then
        emit_report failed "compile gate image build failed" "${case_count}" true
        exit 3
    fi
fi

# Optional explicit scratch wiring: mount a designated disposable scratch
# directory (e.g. the shared folder) and point the container's TMPDIR at it.
# Kept off by default; enabled per scope via PYASC_CANN_GATE_SCRATCH_DIR.
scratch_args=()
if [ -n "${PYASC_CANN_GATE_SCRATCH_DIR:-}" ]; then
    mkdir -p "${PYASC_CANN_GATE_SCRATCH_DIR}"
    scratch_args+=(--volume "${PYASC_CANN_GATE_SCRATCH_DIR}:/scratch:rw"
                   --env "TMPDIR=/scratch")
fi

set +e
timeout --signal=TERM --kill-after=30 "${gate_timeout}" \
    docker run --rm --name "${container_name}" --platform linux/amd64 \
    --volume "${repo_root}:/workspace:ro" \
    --workdir /workspace \
    "${scratch_args[@]}" \
    "${image}" "$@"
rc=$?
set -e

set +e
reconcile_stop "${container_name}" 120
stop_rc=$?
set -e

if [ "${stop_rc}" -eq 1 ]; then
    # FAIL CLOSED: release requires confirmed absence. Emit durable evidence and
    # keep reconciling while retaining the lock; the driver backstop bounds it.
    emit_report failed "compile gate container not confirmed stopped" \
        "${case_count}" false
    echo "[gate] FAIL-CLOSED: ${container_name} not confirmed stopped; retaining lock" >&2
    while :; do
        sleep 5
        set +e
        reconcile_stop "${container_name}" 60
        stop_rc=$?
        set -e
        [ "${stop_rc}" -ne 1 ] && break
        echo "[gate] still reconciling ${container_name}" >&2
    done
fi

if [ "${rc}" -eq 124 ] || [ "${rc}" -eq 137 ]; then
    emit_report failed "compile gate timeout" "${case_count}" true
    exit 124
fi

exit "${rc}"
