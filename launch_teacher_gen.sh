set -ex

NNODES=${NNODES:-1}
NPROC=${NPROC:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12341}
NODE_RANK=${NODE_RANK:-0}
SEED=${SEED:-42}

CKPT_PATH=${CKPT_PATH:-outputs_0321_libero_ood/dreamdojo/exp1201/fold_towel_agilex_3view/checkpoints/iter_000008000/model}
SAVE_ROOT=${SAVE_ROOT:-datasets/teacher_gen_output_fold_towel_agilex_3view}
EMBODIMENT=${EMBODIMENT:-fold_towel_agilex_3view}
EXPERIMENT=${EXPERIMENT:-dreamdojo_2b_480_640_${EMBODIMENT}}

export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FI_EFA_USE_DEVICE_RDMA=1
export RDMAV_FORK_SAFE=1
export TORCH_DIST_INIT_BARRIER=1
# export NCCL_DEBUG=INFO

# CUDA 12 fix: Force PyTorch to use its bundled CUDA libraries
export CUDA_MODULE_LOADING=LAZY
export LD_PRELOAD=""  # Clear any preloaded libraries

echo "Running on $NNODES nodes with $NPROC processes per node. This node rank is $NODE_RANK."

export PYTHONPATH=$(pwd):$PYTHONPATH
export OMP_NUM_THREADS=8
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-./logs}

source .venv/bin/activate

torchrun --nnodes=$NNODES --nproc_per_node=$NPROC \
  --master_port=$MASTER_PORT --master_addr $MASTER_ADDR \
  --node_rank=$NODE_RANK -m cosmos_predict2._src.predict2.action.inference.inference_gr00t_warmup \
  -- \
  --experiment=$EXPERIMENT \
  --ckpt_path $CKPT_PATH \
  --save_root $SAVE_ROOT \
  --embodiment $EMBODIMENT \
  --guidance 0 --chunk_size 12 --start 0 --end 10000 --query_steps 0,9,18,27,34