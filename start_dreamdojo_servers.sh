#!/bin/bash
# Start N DreamDojo servers, each using GPUS_PER_SERVER GPUs (context parallelism).

source .venv/bin/activate
export COSMOS_VERBOSE=1
unset RANK WORLD_SIZE LOCAL_RANK

DD_DIR=${DD_DIR:-"$HOME/workspace/fxz/DreamDojo"}
export HF_HOME=/home/zhiqil/workspace/fxz/hf_cache
# DD_DIR=${DD_DIR:-"$HOME/DreamDojo"}
# export HF_HOME=./hf_cache

CKPT=${CKPT:-"<path_to_model_ema_bf16.pt>"}
EXP=${EXP:-"dreamdojo_2b_480_640_libero"}
SAVE_BASE=${SAVE_BASE:-"$DD_DIR/tmp/dd_results"}
BASE_PORT=${BASE_PORT:-8020}
NUM_SERVERS=${NUM_SERVERS:-5}
GPUS_PER_SERVER=${GPUS_PER_SERVER:-1}
GPU_OFFSET=${GPU_OFFSET:-0}
FPS=${FPS:-10}
STATS_JSON=${STATS_JSON:-""}

for i in $(seq 0 $((NUM_SERVERS - 1))); do
    PORT=$((BASE_PORT + i))
    SAVE_DIR="$SAVE_BASE/server$i"

    GPU_START=$((GPU_OFFSET + i * GPUS_PER_SERVER))
    GPU_LIST=$(seq -s, $GPU_START 1 $((GPU_START + GPUS_PER_SERVER - 1)))

    MASTER_PORT=$((29500 + i))   # each torchrun group needs its own master port

    echo "Starting DreamDojo server $i on GPU(s) $GPU_LIST, port $PORT ..."

    STATS_ARG=""
    if [ -n "$STATS_JSON" ]; then
        STATS_ARG="--stats-json $STATS_JSON"
    fi

    if [ "$GPUS_PER_SERVER" -eq 1 ]; then
        CUDA_VISIBLE_DEVICES=$GPU_LIST python "$DD_DIR/examples/dreamdojo_server.py" \
            --checkpoint "$CKPT" \
            --experiment "$EXP" \
            --save-dir "$SAVE_DIR" \
            --port "$PORT" \
            --fps "$FPS" \
            $STATS_ARG &
    else
        CUDA_VISIBLE_DEVICES=$GPU_LIST \
        torchrun --nproc_per_node="$GPUS_PER_SERVER" --master_port="$MASTER_PORT" \
            "$DD_DIR/examples/dreamdojo_server.py" \
            --checkpoint "$CKPT" \
            --experiment "$EXP" \
            --save-dir "$SAVE_DIR" \
            --port "$PORT" \
            --fps "$FPS" \
            --context-parallel-size "$GPUS_PER_SERVER" \
            $STATS_ARG &
    fi
done

echo "All $NUM_SERVERS servers launched (PIDs: $(jobs -p | tr '\n' ' '))"
echo "Waiting for all servers to be ready..."

ALL_READY=0
while [ $ALL_READY -eq 0 ]; do
    ALL_READY=1
    for i in $(seq 0 $((NUM_SERVERS - 1))); do
        PORT=$((BASE_PORT + i))
        if ! curl -s "http://localhost:$PORT/docs" > /dev/null 2>&1; then
            ALL_READY=0
            break
        fi
    done
    [ $ALL_READY -eq 0 ] && sleep 5
done

echo "All $NUM_SERVERS DreamDojo servers are ready!"
for i in $(seq 0 $((NUM_SERVERS - 1))); do
    GPU_START=$((GPU_OFFSET + i * GPUS_PER_SERVER))
    GPU_END=$((GPU_START + GPUS_PER_SERVER - 1))
    echo "  Server $i -> port $((BASE_PORT + i)), GPU(s) $GPU_START-$GPU_END"
done