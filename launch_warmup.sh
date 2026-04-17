set -ex

NNODES=${NNODES:-1}
NPROC=${NPROC:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12341}
NODE_RANK=${NODE_RANK:-0}
SEED=${SEED:-42}

EXPERIMENT=${EXPERIMENT:-cosmos_predict2p5_2B_action_gr00t_fold_towel_agilex_3view_warmup_no_s3}

# Trainer defaults for warmup experiments.
# Hydra cannot merge experiment DictConfig into structured TrainerConfig attrs,
# so these must be passed as explicit CLI overrides.
TRAINER_MAX_ITER=${TRAINER_MAX_ITER:-20000}
TRAINER_LOGGING_ITER=${TRAINER_LOGGING_ITER:-20}
TRAINER_RUN_VALIDATION=${TRAINER_RUN_VALIDATION:-true}
TRAINER_VALIDATION_ITER=${TRAINER_VALIDATION_ITER:-1000}
TRAINER_MAX_VAL_ITER=${TRAINER_MAX_VAL_ITER:-50}

export WANDB_HTTP_TIMEOUT=300
export WANDB_RETRY_MAX=20
export WANDB_STATS_SAMPLE_RATE_SECONDS=10
export WANDB_STATS_SAMPLES_PER_CORE=1
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
export WANDB_API_KEY=${WANDB_API_KEY:-""}

source .venv/bin/activate

torchrun --nnodes=$NNODES --nproc_per_node=$NPROC \
  --master_port=$MASTER_PORT --master_addr $MASTER_ADDR \
  --node_rank=$NODE_RANK -m scripts.train \
  --config=cosmos_predict2/_src/predict2/interactive/configs/config_warmup.py \
  -- experiment=$EXPERIMENT \
  trainer.max_iter=$TRAINER_MAX_ITER \
  trainer.logging_iter=$TRAINER_LOGGING_ITER \
  trainer.run_validation=$TRAINER_RUN_VALIDATION \
  trainer.validation_iter=$TRAINER_VALIDATION_ITER \
  trainer.max_val_iter=$TRAINER_MAX_VAL_ITER \
  ${EXTRA_ARGS:-}
