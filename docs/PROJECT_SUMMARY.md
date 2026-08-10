# 项目总结（2026-08-07）

## 1. 项目现状

- 流程：LeRobot 生成轨迹 / FIPER pkl → FIPER 做失败检测。
- `/home/zhiyuanjia/lerobot`（旧版）：验证可用，模型 `lerobot/pi05_libero_finetuned`，支持 `--save_fiper_rollouts`，不支持 `--save_trajectories`。
- `/home/zhiyuanjia/lerobot11`（新版）：源码在 `src/lerobot`，必须用 `python -m lerobot.scripts.lerobot_eval`，不能用 `lerobot-eval` 控制台命令（它会指向旧版源码）；支持 `--save_trajectories` 和 `--save_fiper_rollouts`。之前新版遇到 tokenizer 在线下载超时、成功率异常的问题。
- 当前推荐：用旧版 `/home/zhiyuanjia/lerobot` 生成 pkl（已验证 5 条小批量成功率 80%）。

## 2. 生成 FIPER pkl 的命令（旧版）

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
  --save_fiper_rollouts=true \
  --fiper_rollouts_dir=/home/zhiyuanjia/fiper/data/mytask/rollouts/staging_test
```

说明：

- 旧版没有 `--save_trajectories`，只认 `--save_fiper_rollouts` 和 `--fiper_rollouts_dir`。
- 生成后把 pkl 移入 `rollouts/test/` 或 `rollouts/calibration/`，文件名形式 `episode_XXX_success.pkl` / `episode_XXX_fail.pkl`。

## 3. lerobot11 新版命令（备用）

```bash
cd /home/zhiyuanjia/lerobot11
export PYTHONPATH=/home/zhiyuanjia/lerobot11/src
export MUJOCO_GL=osmesa
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
/home/zhiyuanjia/miniconda3/envs/lerobot/bin/python -m lerobot.scripts.lerobot_eval \
  --policy.path=<模型路径> \
  --env.type=libero \
  --env.task=libero_10 \
  --env.task_ids=[7] \
  --eval.n_episodes=50 \
  --eval.batch_size=1 \
  --save_trajectories=true \
  --save_fiper_rollouts=true \
  --fiper_rollouts_dir=/path/to/output
```

注意：跑 lerobot11 时不要敲 `lerobot-eval`，必须用 `python -m lerobot.scripts.lerobot_eval`。

## 4. FIPER pkl 格式

每个 rollout 存为一个 pkl：

```python
{
  "metadata": {"successful": bool, "episode_id": int, ...},
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

## 5. FIPER 训练 / 评估 / 出图

配置文件：`/home/zhiyuanjia/fiper/configs/default.yaml`

- 当前设置：`tasks: ["mytask"]`，`rnd_models: ["rnd_oe"]`，`methods: ["entropy", "logpzo"]`，`combine_methods: True`（生成 `rnd_oe_and_entropy`），`train_rnd: True`。

训练 + 评估：

```bash
conda activate fiper
cd /home/zhiyuanjia/fiper
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
python scripts/run_fiper.py
```

汇总 / 绘图：

```bash
cd /home/zhiyuanjia/fiper
python scripts/results_generation.py
```

绘图开关在 `/home/zhiyuanjia/fiper/configs/results/base.yaml`，包括 `uncertainty_plots`、`quantile_impact`、`rollout_type_stats` 等的 `create_plots`。

失败检测警告视频（视频中叠加 rnd-oe 原始分 / 归一化分以及失败判定）：

```bash
cd /home/zhiyuanjia/fiper
python scripts/generate_warning_videos.py \
  --method rnd_oe \
  --task mytask \
  --threshold_style tvt_cp_band \
  --quantile 0.9 \
  --window 45 \
  --fps 10
```

## 6. 当前数据（2026-08-09 划分后）

- `/home/zhiyuanjia/fiper/data/mytask/rollouts/test/`：70 条（42 成功 / 28 失败），已校验格式正确。
- `/home/zhiyuanjia/fiper/data/mytask/rollouts/calibration/`：30 条，全部成功（来自新生成 50 条中的 30 条成功轨迹）。
- `/home/zhiyuanjia/fiper/data/mytask/rollouts/calibration_old_2026_08_backup/`：22 条旧 cal 备份（确认后可删除或并入 test）。
- 已删除：`staging_cal_new/`、所有 `staging_*` 临时目录、`/home/zhiyuanjia/lerobot/rollouts`（3.8GB 重复副本）、`lerobot/outputs/eval/2026-08-07`、`lerobot11/outputs/eval/2026-08-06`、`lerobot11/logs/gen_calib_50.log`、重复的 `FIPER_USAGE.md`，以及 Windows 工作区中的临时脚本和下载文件。
- `lerobot/outputs/eval/2026-05-31`、`2026-06-01` 为项目原有输出，未动。

## 7. 下次待办

1. `configs/default.yaml` 已配置完成：只训练 mytask，目标方法为 `rnd_oe_and_entropy`（rnd_oe AND entropy）和 `logpzo`。训练前先删除旧 rnd_oe checkpoint，再跑 `run_fiper.py`（命令见上文第 5 节）。
2. 当前数据已满足 test=70 / cal=30；如再生成，用旧版命令，cal 只保留成功轨迹。
3. 跑 `run_fiper.py` 后运行 `results_generation.py` 出图和警告视频。
