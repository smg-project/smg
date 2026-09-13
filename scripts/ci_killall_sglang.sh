#!/bin/bash

if [ "$1" = "rocm" ]; then
    echo "Running in ROCm mode"

    # Clean SGLang processes
    pgrep -f 'sglang::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt|sgl_diffusion::' | xargs -r kill -9

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
    if [ $# -gt 0 ]; then
        sleep 2
        if ! DIRTY_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null); then
            echo "::error::Unable to query GPU memory on node '$NODE_INFO'."
            exit 1
        fi
        DIRTY_GPUS=$(echo "$DIRTY_GPUS" | awk -F', ' '$2 > 100 {print "GPU " $1 ": " $2 " MiB used"}')
        if [ -n "$DIRTY_GPUS" ]; then
            echo "::error::GPU not clean on node '$NODE_INFO':"
            echo "$DIRTY_GPUS"
            exit 1
        fi
    fi
fi
