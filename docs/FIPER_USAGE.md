# FIPER Usage

本文档说明从 LeRobot 生成轨迹 pkl 到 FIPER 分析、出图的完整用法。

## 1. 生成 FIPER pkl

使用 `lerobot`，用 `python -m lerobot.scripts.lerobot_eval`，或者用 `lerobot-eval` 控制台命令

```bash
cd /home/zhiyuanjia/lerobot
export PYTHONPATH=/home/zhiyuanjia/lerobot/src
export MUJOCO_GL=osmesa
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
/home/zhiyuanjia/miniconda3/envs/lerobot/bin/python -m lerobot.scripts.lerobot_eval \
  --policy.path=lerobot/pi05_libero_finetuned \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=false \
  --env.type=libero \
  --env.task=libero_10 \
  --env.task_ids=[7] \
  --eval.n_episodes=50 \
  --eval.batch_size=1 \
  --save_trajectories=true \
  --save_fiper_rollouts=true \
  --fiper_rollouts_dir=/home/zhiyuanjia/fiper/data/mytask/rollouts/staging_test
```

参数说明：

- `--save_trajectories=true`：额外保存 HDF5 轨迹，可选。
- `--save_fiper_rollouts=true` + `--fiper_rollouts_dir=...`：保存 FIPER pkl。
- 生成后把 pkl 按用途放入 `rollouts/test/`、`rollouts/calibration/` 等目录。

也可以直接运行仓库里的脚本：

```bash
bash lerobot/eval_commands.sh
```

## 2. FIPER pkl 格式

每个 rollout 是一个 pkl：

```python
{
  "metadata": {"successful": bool, "episode_id": int},
  "rollout": [
    {
      "action":        np.ndarray (7,),          # 当前执行动作
      "action_pred":   np.ndarray (50, 7),       # 预测动作块
      "obs_embedding": np.ndarray (2048,),       # 观测编码
      "agent_pos":     np.ndarray (34,),         # 机器人状态
      "rgb":           np.ndarray (360, 360, 3)  # uint8 图像
    },
    ...
  ]
}
```

文件名：`episode_XXX_success.pkl`（成功）或 `episode_XXX_fail.pkl`（失败）。

## 3. FIPER 训练与评估

配置在 `configs/default.yaml`：

- `tasks: ["mytask"]`
- `rnd_models: ["rnd_oe"]`
- `methods: ["entropy", "logpzo"]`
- `combine_methods: True`，组合出 `rnd_oe_and_entropy`
- `train_rnd: True`

运行：

```bash
conda activate fiper
cd /home/zhiyuanjia/fiper
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
python scripts/run_fiper.py
```

## 4. 生成结果与视频

```bash
cd /home/zhiyuanjia/fiper
python scripts/results_generation.py
```

行为由 `configs/results/base.yaml` 控制：

```yaml
extract_warning_frames:
  create_plots: True
  method: "rnd_oe_and_entropy"   # 或 "logpzo" 等
  max_episodes: 10               # 最多生成多少条视频
  only_fail: True                # 只取失败轨迹
```

输出位置：

```text
data/results/videos_with_warnings/mytask/<method>/rollouts/episode_XXX.mp4
```

同一个 `method` 目录下还有 `test_frames/` 中间帧。要对比不同方法，把 `method` 换成对应名字再跑一次即可。

## 5. 其他方法

`implemented_methods` 包括：`rnd_oe`、`rnd_a`、`entropy`、`tc`、`similarity`、`logpzo`。改 `default.yaml` 里的 `methods` / `rnd_models` 即可训练和评估其他方法。
