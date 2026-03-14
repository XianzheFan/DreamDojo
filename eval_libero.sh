export PYTHONPATH=/root/workspace/fxz/DreamDojo:$PYTHONPATH
export COSMOS_VERBOSE=1
source ~/workspace/fxz/DreamDojo/.venv/bin/activate

torchrun --nproc_per_node=8 examples/action_conditioned.py \
  -o outputs/action_conditioned/libero \
  --checkpoints-dir outputs_0312/dreamdojo/exp1201/libero/checkpoints \
  --experiment dreamdojo_2b_480_640_libero \
  --save-dir outputs_0312/dreamdojo/exp1201/libero/eval_results \
  --num-frames 13 \
  --num-samples 100 \
  --dataset-path datasets/libero_object_dreamdojo \
  --data-split test \
  --deterministic-uniform-sampling \
  --checkpoint-interval 10000 \
  --context-parallel-size 1
