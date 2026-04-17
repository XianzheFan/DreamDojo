set -ex

NNODES=${NNODES:-1}
NPROC=${NPROC:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12341}
NODE_RANK=${NODE_RANK:-0}
SEED=${SEED:-42}

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
export WANDB_API_KEY=${WANDB_API_KEY:?Please set WANDB_API_KEY environment variable}

source .venv/bin/activate

EXPERIMENT=${EXPERIMENT:-cosmos_predict2p5_2B_action_gr00t_fold_towel_agilex_3view_self_forcing_no_s3}
CKPT_PATH=${CKPT_PATH:-checkpoints/iter_000003000}
INPUT_JSON=${INPUT_JSON:-datasets/eval/info.json}

torchrun --nproc_per_node=$NPROC -m cosmos_predict2._src.predict2.interactive.inference.action_video2world \
  --config=cosmos_predict2/_src/predict2/interactive/configs/config_distill.py \
  --experiment=$EXPERIMENT \
  --ckpt_path $CKPT_PATH \
  --input_json $INPUT_JSON \
  "$@"