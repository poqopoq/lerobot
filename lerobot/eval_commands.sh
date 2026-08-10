#!/usr/bin/env bash
# Generate LIBERO trajectories + FIPER pkl using lerobot11.
set -e

PYTHON_BIN=${PYTHON_BIN:-/home/zhiyuanjia/miniconda3/envs/lerobot/bin/python}
LEROBOT_ROOT=${LEROBOT_ROOT:-/home/zhiyuanjia/lerobot11}
OUT_DIR=${OUT_DIR:-/home/zhiyuanjia/fiper/data/mytask/rollouts/staging_test}
N_EPISODES=${N_EPISODES:-50}

export PYTHONPATH="$LEROBOT_ROOT/src"
export MUJOCO_GL=osmesa
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$PYTHON_BIN" -m lerobot.scripts.lerobot_eval \
  --policy.path=lerobot/pi05_libero_finetuned \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=false \
  --env.type=libero \
  --env.task=libero_10 \
  --env.task_ids=[7] \
  --eval.n_episodes="$N_EPISODES" \
  --eval.batch_size=1 \
  --save_trajectories=true \
  --save_fiper_rollouts=true \
  --fiper_rollouts_dir="$OUT_DIR"
