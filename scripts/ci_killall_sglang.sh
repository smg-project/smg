#!/bin/bash

if [ "$1" = "rocm" ]; then
    echo "Running in ROCm mode"

    # Clean SGLang processes
    pgrep -f 'sglang::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt|sgl_diffusion::' | xargs -r kill -9

    # `rocm nuke_gpus`: also kill every process holding the GPUs, like nuke_gpus
    # below. On ROCm every GPU user holds /dev/kfd open. They are found in /proc,
    # as vLLM's ROCm image has neither lsof nor fuser.
    kfd_holders() {
        local fd
        for fd in /proc/[0-9]*/fd/*; do
            if [ "$(readlink "$fd" 2>/dev/null)" = /dev/kfd ]; then
                fd=${fd#/proc/}
                echo "${fd%%/*}"
            fi
        done | sort -u
    }
    if [ $# -gt 1 ]; then
        kfd_holders | xargs -r kill -9 2>/dev/null
        sleep 2
        echo "Processes still holding /dev/kfd: $(kfd_holders | wc -l)"
    fi

else
    # Show current GPU status
    nvidia-smi

    # Clean SGLang processes
    pgrep -f 'sglang::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt|sgl_diffusion::' | xargs -r kill -9

    # Clean all GPU processes if any argument is provided
    if [ $# -gt 0 ]; then
        # Install lsof if not already available
        if ! command -v lsof >/dev/null 2>&1; then
            if command -v sudo >/dev/null 2>&1; then
                bash "$(dirname "${BASH_SOURCE[0]}")/ci_apt_mirror.sh"
                sudo apt-get update
                sudo apt-get install -y lsof
            else
                bash "$(dirname "${BASH_SOURCE[0]}")/ci_apt_mirror.sh"
                apt-get update
                apt-get install -y lsof
            fi
        fi
        kill -9 $(nvidia-smi | sed -n '/Processes:/,$p' | grep "   [0-9]" | awk '{print $5}') 2>/dev/null
        lsof /dev/nvidia* | awk '{print $2}' | xargs kill -9 2>/dev/null
    fi

    # Show GPU status after clean up
    nvidia-smi

    NODE_INFO="${NODE_IP:-$(hostname)}"
    echo "Running on node: $NODE_INFO"

    # The processes this container can see that hold a GPU device file open,
    # one pid per line. Reads $CI_GPU_GUARD_PROC_ROOT (/proc; the tests point
    # it at a scratch tree).
    gpu_device_holders() {
        local fd pid
        for fd in "${CI_GPU_GUARD_PROC_ROOT:-/proc}"/[0-9]*/fd/*; do
            case "$(readlink "$fd" 2>/dev/null)" in
                /dev/nvidia*)
                    pid=${fd%/fd/*}
                    echo "${pid##*/}" ;;
            esac
        done | sort -un
    }

    # The command line of a process this container can see.
    gpu_command_of() {
        tr '\0' ' ' < "${CI_GPU_GUARD_PROC_ROOT:-/proc}/$1/cmdline" 2>/dev/null | sed 's/ $//'
    }

    # " E2E_JOB_MARKER=<value>" when the process carries a job's marker (the
    # job that started it; see ci_reap_job_processes.py), else nothing.
    gpu_job_marker_of() {
        local marker
        marker=$(tr '\0' '\n' < "${CI_GPU_GUARD_PROC_ROOT:-/proc}/$1/environ" 2>/dev/null | grep -m1 '^E2E_JOB_MARKER=' || true)
        if [ -n "$marker" ]; then
            echo " $marker"
        fi
    }

    # Who holds the cards, from every view this container has; sets GPU_HOLDERS
    # to a short summary for the verdict line. The driver's compute-app list is
    # the list behind the process table above, its pids translated into this
    # container's pid namespace and the ones it cannot translate dropped: from
    # inside a pod it is usually empty beside a full card (which is why the
    # table reads "No running processes found"), and the fallback text then
    # names the holder; a pid it does list that this /proc lacks runs in
    # another pid namespace. The processes of this container holding a device
    # file are named with the job marker they carry, and the container
    # runtime's view is added when a docker daemon is reachable.
    gpu_holders() {
        local root map apps idx pid name mem view containers
        root=${CI_GPU_GUARD_PROC_ROOT:-/proc}
        GPU_HOLDERS=
        echo "gpu guard: this job: runner '${RUNNER_NAME:-unknown}', marker '${E2E_JOB_MARKER:-none}'"
        map=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits 2>/dev/null || true)
        apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
        if [ -z "$apps" ]; then
            echo "gpu guard: the driver lists no compute process on any GPU"
        fi
        while IFS=$'\t' read -r idx pid name mem; do
            [ -n "$pid" ] || continue
            if [ -d "$root/$pid" ]; then
                view="in this job's view: $(gpu_command_of "$pid")$(gpu_job_marker_of "$pid")"
                GPU_HOLDERS="${GPU_HOLDERS:+$GPU_HOLDERS, }pid $pid"
            else
                view="not in this job's pid namespace"
            fi
            echo "gpu guard: GPU $idx: pid $pid ($name) holds $mem MiB; $view"
        done <<< "$(echo "$apps" | awk -F', ' -v map="$map" '
            BEGIN { n = split(map, rows, "\n"); for (i = 1; i <= n; i++) { split(rows[i], f, ", "); index_of[f[2]] = f[1] } }
            NF >= 4 { print (($1 in index_of) ? index_of[$1] : "?") "\t" $2 "\t" $3 "\t" $4 }')"
        while read -r pid; do
            [ -n "$pid" ] || continue
            echo "gpu guard: pid $pid holds a GPU device file: $(gpu_command_of "$pid")$(gpu_job_marker_of "$pid")"
            case ", $GPU_HOLDERS," in
                *" pid $pid,"*) ;;
                *) GPU_HOLDERS="${GPU_HOLDERS:+$GPU_HOLDERS, }pid $pid" ;;
            esac
        done <<< "$(gpu_device_holders)"
        if command -v docker >/dev/null 2>&1; then
            containers=$(timeout 10 docker ps --format '{{.ID}} {{.Image}} {{.Names}} up {{.RunningFor}}' 2>/dev/null | paste -sd ';' - || true)
            echo "gpu guard: containers of this runner's docker: ${containers:-none}"
        fi
        GPU_HOLDERS=${GPU_HOLDERS:-"a process outside this job's view (another pid namespace: a previous job's engine still exiting, or a leaked context)"}
    }

    if [ $# -gt 0 ]; then
        # The verdict. A card still holding memory here has been, in every
        # sighting so far, a previous job's engine that had not finished dying:
        # a cancelled job's pod gets minutes of termination grace, and its
        # processes are invisible from this container. So the cards are re-read
        # every CI_GPU_GUARD_POLL_SECS for up to CI_GPU_GUARD_WAIT_SECS (0: one
        # reading) before the job is failed; every reading is printed, and so
        # is whoever holds the memory, as far as this container can see, at the
        # first dirty reading and at the verdict. The threshold is 100 MiB.
        WAIT_SECS=${CI_GPU_GUARD_WAIT_SECS:-300}
        POLL_SECS=${CI_GPU_GUARD_POLL_SECS:-10}
        case "$WAIT_SECS$POLL_SECS" in
            *[!0-9]*)
                echo "::warning::CI_GPU_GUARD_WAIT_SECS and CI_GPU_GUARD_POLL_SECS take whole seconds; using 300 and 10."
                WAIT_SECS=300
                POLL_SECS=10 ;;
        esac
        sleep 2
        SECONDS=0
        REPORTED_AT=
        while :; do
            if ! USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null); then
                echo "::error::Unable to query GPU memory on node '$NODE_INFO'."
                exit 1
            fi
            DIRTY_GPUS=$(echo "$USED" | awk -F', ' '$2 > 100 {print "GPU " $1 ": " $2 " MiB used"}')
            if [ -z "$DIRTY_GPUS" ]; then
                if [ -n "$REPORTED_AT" ]; then
                    echo "gpu guard: every GPU clean after ${SECONDS}s"
                fi
                break
            fi
            echo "gpu guard: ${SECONDS}s: $(echo "$DIRTY_GPUS" | paste -sd ';' - | sed 's/;/; /g')"
            if [ -z "$REPORTED_AT" ]; then
                gpu_holders
                REPORTED_AT=$SECONDS
            fi
            if [ "$SECONDS" -ge "$WAIT_SECS" ]; then
                if [ "$SECONDS" -gt "$REPORTED_AT" ]; then
                    gpu_holders
                fi
                echo "::error::GPU not clean on node '$NODE_INFO' after ${SECONDS}s, held by $GPU_HOLDERS:"
                echo "$DIRTY_GPUS"
                exit 1
            fi
            sleep "$POLL_SECS"
        done
    fi
fi
