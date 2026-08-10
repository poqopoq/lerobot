#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
lerobot-eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
lerobot-eval \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
"""

import concurrent.futures as cf
import json
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Any, TypedDict

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import (
    add_envs_task,
    check_env_attributes_and_types,
    close_envs,
    preprocess_observation,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.types import PolicyAction
from lerobot.utils.constants import ACTION, DONE, OBS_STR, REWARD
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    init_logging,
    inside_slurm,
)
from .fiper_utils import FIPERTrajectoryCollector, save_fiper_rollouts_batch
from .trajectory_to_hdf5 import batch_save_episodes_to_hdf5


def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
    fiper_collector: "FIPERTrajectoryCollector | None" = None,
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
        render_callback: Optional rendering callback to be used after the environments are reset, and after
            every step.
    Returns:
        The dictionary described above.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    # Reset the policy and environments.
    policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    all_observations = []
    all_raw_observations = []  # Track raw obs before preprocessing (for LIBERO format)
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []
    all_fiper_step_data = []  # per-step FIPER extras: action_pred, obs_embedding

    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        leave=False,
    )
    check_env_attributes_and_types(env)
    while not np.all(done) and step < max_steps:
        # Collect raw observations BEFORE preprocessing (preserves LIBERO format keys)
        if return_observations:
            # Try to get raw LIBERO observations from environment if available
            libero_raw_obs_list = []
            for env_idx in range(env.num_envs):
                if hasattr(env.envs[env_idx], '_last_libero_raw_obs'):
                    libero_raw_obs_list.append(deepcopy(env.envs[env_idx]._last_libero_raw_obs))
                else:
                    libero_raw_obs_list.append(None)
            
            # Store LIBERO raw obs if available, otherwise fall back to formatted obs
            if any(obs is not None for obs in libero_raw_obs_list):
                # Stack LIBERO raw observations (handle per-env differences)
                libero_obs_dict = {}
                for obs_key in ['agentview_rgb', 'eye_in_hand_rgb', 'ee_pos', 'ee_ori', 'ee_states', 'gripper_states', 'joint_states']:
                    obs_list = []
                    for libero_obs in libero_raw_obs_list:
                        if libero_obs is not None and obs_key in libero_obs and libero_obs[obs_key] is not None:
                            obs_list.append(libero_obs[obs_key])
                    if obs_list:
                        libero_obs_dict[obs_key] = np.stack(obs_list) if len(obs_list) == env.num_envs else obs_list[0]
                
                all_raw_observations.append(libero_obs_dict if libero_obs_dict else deepcopy(observation))
            else:
                all_raw_observations.append(deepcopy(observation))
        
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(observation)
        if return_observations:
            all_observations.append(deepcopy(observation))

        # Infer "task" from attributes of environments.
        # TODO: works with SyncVectorEnv but not AsyncVectorEnv
        observation = add_envs_task(env, observation)

        # Apply environment-specific preprocessing (e.g., LiberoProcessorStep for LIBERO)
        observation = env_preprocessor(observation)

        observation = preprocessor(observation)
        with torch.inference_mode():
            action = policy.select_action(observation)
        action = postprocessor(action)

        # Capture FIPER per-step extras (action_pred, obs_embedding) right after select_action.
        if fiper_collector is not None:
            action_pred, obs_embedding = fiper_collector.get_step_extras()
            all_fiper_step_data.append({"action_pred": action_pred, "obs_embedding": obs_embedding})

        action_transition = {ACTION: action}
        action_transition = env_postprocessor(action_transition)
        action = action_transition[ACTION]

        # Convert to CPU / numpy.
        action_numpy: np.ndarray = action.to("cpu").numpy()
        assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

        # Apply the next action.
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if render_callback is not None:
            render_callback(env)

        # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
        # available if none of the envs finished.
        if "final_info" in info:
            final_info = info["final_info"]
            if not isinstance(final_info, dict):
                raise RuntimeError(
                    "Unsupported `final_info` format: expected dict (Gymnasium >= 1.0). "
                    "You're likely using an older version of gymnasium (< 1.0). Please upgrade."
                )
            successes = final_info["is_success"].tolist()
        else:
            successes = [False] * env.num_envs

        # Keep track of which environments are done so far.
        # Mark the episode as done if we reach the maximum step limit.
        # This ensures that the rollout always terminates cleanly at `max_steps`,
        # and allows logging/saving (e.g., videos) to be triggered consistently.
        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        step += 1
        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
        )
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()

    # Track the final observation.
    if return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(deepcopy(observation))

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    if return_observations:
        def stack_nested_observations(obs_list):
            """Recursively stack nested observation dictionaries."""
            if not obs_list or not obs_list[0]:
                return {}
            
            stacked = {}
            for key in obs_list[0]:
                values = [obs[key] for obs in obs_list]
                
                # Check if value is a dict (nested structure)
                if isinstance(values[0], dict):
                    # Recursively stack nested dict
                    stacked[key] = stack_nested_observations(values)
                elif isinstance(values[0], torch.Tensor):
                    # Stack tensor values
                    stacked[key] = torch.stack(values, dim=1)
                else:
                    # Try to convert to tensor and stack
                    try:
                        stacked[key] = torch.stack([torch.as_tensor(v) for v in values], dim=1)
                    except Exception as e:
                        # Skip values that can't be stacked
                        print(f"Warning: Could not stack observation '{key}': {e}")
                        continue
            
            return stacked
        
        stacked_observations = stack_nested_observations(all_observations)
        ret[OBS_STR] = stacked_observations
        
        # Also stack raw observations (before preprocessing) for LIBERO format
        if all_raw_observations:
            stacked_raw_observations = stack_nested_observations(all_raw_observations)
            ret["raw_obs"] = stacked_raw_observations

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    if all_fiper_step_data:
        ret["fiper_step_data"] = all_fiper_step_data

    return ret


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    save_trajectories: bool = True,
    trajectories_dir: Path | None = None,
    save_fiper_rollouts: bool = False,
    fiper_rollouts_dir: Path | None = None,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
        max_episodes_rendered: Maximum number of episodes to render into videos.
        videos_dir: Where to save rendered videos.
        return_episode_data: Whether to return episode data for online training. Incorporates the data into
            the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout. For all subsequent rollouts the
            seed is incremented by 1. If not provided, the environments are not manually seeded.
        save_trajectories: Whether to save episode trajectories to HDF5 format (for FAIL-Detect analysis).
        trajectories_dir: Directory where to save trajectories (HDF5 files). If not provided, uses videos_dir.
        save_fiper_rollouts: Whether to save FIPER-format .pkl rollouts (separate from HDF5 trajectories).
        fiper_rollouts_dir: Directory where to save FIPER .pkl rollouts.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    if save_trajectories and not trajectories_dir:
        # Default to videos_dir if not specified
        if videos_dir:
            trajectories_dir = Path(videos_dir).parent / "trajectories"
        else:
            trajectories_dir = Path("trajectories")
    
    trajectories_dir = Path(trajectories_dir) if trajectories_dir else None

    # Set up FIPER rollout saving (separate from HDF5 trajectories)
    if save_fiper_rollouts and not fiper_rollouts_dir:
        fiper_rollouts_dir = Path(videos_dir).parent / "fiper_rollouts" if videos_dir else Path("fiper_rollouts")
    fiper_rollouts_dir = Path(fiper_rollouts_dir) if fiper_rollouts_dir else None
    if fiper_rollouts_dir:
        fiper_rollouts_dir.mkdir(parents=True, exist_ok=True)

    # Create FIPER collector to capture action_pred and obs_embedding during rollouts
    fiper_collector = FIPERTrajectoryCollector(policy) if save_fiper_rollouts else None

    if not isinstance(policy, PreTrainedPolicy):
        exc = ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )
        try:
            from peft import PeftModel

            if not isinstance(policy, PeftModel):
                raise exc
        except ImportError:
            raise exc from None

    start = time.time()
    policy.eval()

    # Determine how many batched rollouts we need to get n_episodes. Note that if n_episodes is not evenly
    # divisible by env.num_envs we end up discarding some data in the last batch.
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    # Keep track of some metrics.
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    threads = []  # for video saving threads
    n_episodes_rendered = 0  # for saving the correct number of videos

    # Callback for visualization.
    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render_now)]))  # noqa: B023
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            # Here we must render all frames and discard any we don't need.
            ep_frames.append(np.stack(env.call("render")[:n_to_render_now]))

    if max_episodes_rendered > 0:
        video_paths: list[str] = []
    
    trajectory_paths: list[str] = [] if save_trajectories and trajectories_dir else []
    fiper_rollout_paths: list[str] = [] if save_fiper_rollouts and fiper_rollouts_dir else []

    if return_episode_data:
        episode_data: dict | None = None

    # we dont want progress bar when we use slurm, since it clutters the logs
    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        # Cache frames for rendering videos. Each item will be (b, h, w, c), and the list indexes the rollout
        # step.
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            )
        rollout_data = rollout(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data or save_trajectories or save_fiper_rollouts,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
            fiper_collector=fiper_collector,
        )

        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        n_steps = rollout_data["done"].shape[1]
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        # Extend metrics.
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.append(None)

        # FIXME: episode_data is either None or it doesn't exist
        if return_episode_data:
            this_episode_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * env.num_envs,
                start_data_index=(0 if episode_data is None else (episode_data["index"][-1].item() + 1)),
                fps=env.unwrapped.metadata["render_fps"],
            )
            if episode_data is None:
                episode_data = this_episode_data
            else:
                # Some sanity checks to make sure we are correctly compiling the data.
                assert episode_data["episode_index"][-1] + 1 == this_episode_data["episode_index"][0]
                assert episode_data["index"][-1] + 1 == this_episode_data["index"][0]
                # Concatenate the episode data (handle nested dicts).
                def concatenate_episode_data_recursive(data1, data2):
                    """Recursively concatenate nested observation dictionaries."""
                    result = {}
                    for k in data1:
                        if isinstance(data1[k], dict) and isinstance(data2[k], dict):
                            # Recursively concatenate nested dicts
                            result[k] = concatenate_episode_data_recursive(data1[k], data2[k])
                        elif isinstance(data1[k], torch.Tensor) and isinstance(data2[k], torch.Tensor):
                            # Concatenate tensors
                            result[k] = torch.cat([data1[k], data2[k]])
                        else:
                            # Default: use data1 (shouldn't happen)
                            result[k] = data1[k]
                    return result
                
                episode_data = concatenate_episode_data_recursive(episode_data, this_episode_data)

        # Maybe render video for visualization.
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, *)
            for stacked_frames, done_index in zip(
                batch_stacked_frames, done_indices.flatten().tolist(), strict=False
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],  # + 1 to capture the last observation
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1
        
        # Save trajectories to HDF5 for FAIL-Detect analysis.
        if save_trajectories and trajectories_dir:
            batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any").tolist()
            saved_paths = batch_save_episodes_to_hdf5(
                rollout_data=rollout_data,
                done_indices=done_indices,
                successes=batch_successes,
                start_episode_idx=batch_ix * env.num_envs,
                output_dir=trajectories_dir,
                max_steps_per_episode=n_steps,
                save_raw_obs="raw_obs" in rollout_data,  # Pass raw_obs if available
            )
            trajectory_paths.extend(saved_paths)

        # Save FIPER-format .pkl rollouts (separate from HDF5 trajectories).
        if save_fiper_rollouts and fiper_rollouts_dir:
            batch_successes_fiper = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any").tolist()
            fiper_step_data = rollout_data.get("fiper_step_data", [])
            obs_for_fiper = rollout_data.get(OBS_STR, {})
            saved_pkl_paths = save_fiper_rollouts_batch(
                rollout_data={**rollout_data, "observation": obs_for_fiper},
                fiper_step_data=fiper_step_data,
                done_indices=done_indices,
                successes=batch_successes_fiper,
                start_episode_idx=batch_ix * env.num_envs,
                output_dir=fiper_rollouts_dir,
            )
            fiper_rollout_paths.extend(saved_pkl_paths)

        progbar.set_postfix(
            {"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}
        )

    # Wait till all video rendering threads are done.
    for thread in threads:
        thread.join()

    # Clean up FIPER collector hooks.
    if fiper_collector is not None:
        fiper_collector.cleanup()

    # Compile eval info.
    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
            }
            for i, (sum_reward, max_reward, success, seed) in enumerate(
                zip(
                    sum_rewards[:n_episodes],
                    max_rewards[:n_episodes],
                    all_successes[:n_episodes],
                    all_seeds[:n_episodes],
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }

    if return_episode_data:
        info["episodes"] = episode_data

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths
    
    if trajectory_paths:
        info["trajectory_paths"] = [str(p) for p in trajectory_paths]

    if fiper_rollout_paths:
        info["fiper_rollout_paths"] = [str(p) for p in fiper_rollout_paths]

    return info


def _compile_episode_data(
    rollout_data: dict, done_indices: Tensor, start_episode_index: int, start_data_index: int, fps: float
) -> dict:
    """Convenience function for `eval_policy(return_episode_data=True)`

    Compiles all the rollout data into a Hugging Face dataset.

    Similar logic is implemented when datasets are pushed to hub (see: `push_to_hub`).
    """
    ep_dicts = []
    total_frames = 0
    for ep_ix in range(rollout_data[ACTION].shape[0]):
        # + 2 to include the first done frame and the last observation frame.
        num_frames = done_indices[ep_ix].item() + 2
        total_frames += num_frames

        # Here we do `num_frames - 1` as we don't want to include the last observation frame just yet.
        ep_dict = {
            ACTION: rollout_data[ACTION][ep_ix, : num_frames - 1],
            "episode_index": torch.tensor([start_episode_index + ep_ix] * (num_frames - 1)),
            "frame_index": torch.arange(0, num_frames - 1, 1),
            "timestamp": torch.arange(0, num_frames - 1, 1) / fps,
            DONE: rollout_data["done"][ep_ix, : num_frames - 1],
            "next.success": rollout_data["success"][ep_ix, : num_frames - 1],
            REWARD: rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),
        }

        # For the last observation frame, all other keys will just be copy padded.
        for k in ep_dict:
            ep_dict[k] = torch.cat([ep_dict[k], ep_dict[k][-1:]])

        def extract_nested_obs(obs_dict):
            """Recursively extract episode observations from stacked data."""
            result = {}
            for key, value in obs_dict.items():
                if isinstance(value, dict):
                    # Recursively extract from nested dict
                    result[key] = extract_nested_obs(value)
                else:
                    # Extract episode data from tensor
                    result[key] = value[ep_ix, :num_frames]
            return result
        
        nested_obs = extract_nested_obs(rollout_data[OBS_STR])
        ep_dict.update(nested_obs)

        ep_dicts.append(ep_dict)

    data_dict = {}
    
    def concatenate_nested_data(list_of_dicts):
        """Recursively concatenate nested observation dictionaries."""
        result = {}
        for key in list_of_dicts[0]:
            values = [d[key] for d in list_of_dicts]
            if isinstance(values[0], dict):
                # Recursively concatenate nested dict
                result[key] = concatenate_nested_data(values)
            else:
                # Concatenate tensor values
                result[key] = torch.cat(values)
        return result
    
    for key in ep_dicts[0]:
        values = [x[key] for x in ep_dicts]
        if isinstance(values[0], dict):
            # Recursively concatenate nested observations
            data_dict[key] = concatenate_nested_data(values)
        else:
            # Concatenate tensor values
            data_dict[key] = torch.cat(values)

    data_dict["index"] = torch.arange(start_data_index, start_data_index + total_frames, 1)

    return data_dict


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making environment.")
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )

    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )

    policy.eval()

    # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    # Create environment-specific preprocessor and postprocessor (e.g., for LIBERO environments)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    # Prepare trajectories directory if saving
    trajectories_dir = None
    if cfg.save_trajectories:
        if cfg.trajectories_dir:
            trajectories_dir = Path(cfg.trajectories_dir)
        else:
            trajectories_dir = Path(cfg.output_dir) / "trajectories"
        trajectories_dir.mkdir(parents=True, exist_ok=True)

    # Prepare FIPER rollouts directory if saving
    fiper_rollouts_dir = None
    if cfg.save_fiper_rollouts:
        if cfg.fiper_rollouts_dir:
            fiper_rollouts_dir = Path(cfg.fiper_rollouts_dir)
        else:
            fiper_rollouts_dir = Path(cfg.output_dir) / "fiper_rollouts"
        fiper_rollouts_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=10,
            videos_dir=Path(cfg.output_dir) / "videos",
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
            save_trajectories=cfg.save_trajectories,
            trajectories_dir=trajectories_dir,
            return_episode_data=cfg.save_trajectories,  # Collect obs when saving trajectories
            save_fiper_rollouts=cfg.save_fiper_rollouts,
            fiper_rollouts_dir=fiper_rollouts_dir,
        )
        print("Overall Aggregated Metrics:")
        print(info["overall"])

        # Print per-suite stats
        for task_group, task_group_info in info.items():
            print(f"\nAggregated Metrics for {task_group}:")
            print(task_group_info)
    # Close all vec envs
    close_envs(envs)

    # Save info
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    logging.info("End of eval")


# ---- typed payload returned by one task eval ----
class TaskMetrics(TypedDict):
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    video_paths: list[str]
    trajectory_paths: list[str]
    fiper_rollout_paths: list[str]


ACC_KEYS = ("sum_rewards", "max_rewards", "successes", "video_paths", "trajectory_paths", "fiper_rollout_paths")


def eval_one(
    env: gym.vector.VectorEnv,
    *,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
    save_trajectories: bool = True,
    trajectories_dir: Path | None = None,
    save_fiper_rollouts: bool = False,
    fiper_rollouts_dir: Path | None = None,
) -> TaskMetrics:
    """Evaluates one task_id of one suite using the provided vec env."""

    task_videos_dir = videos_dir
    task_trajectories_dir = trajectories_dir

    task_result = eval_policy(
        env=env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        save_trajectories=save_trajectories,
        trajectories_dir=task_trajectories_dir,
        save_fiper_rollouts=save_fiper_rollouts,
        fiper_rollouts_dir=fiper_rollouts_dir,
    )

    per_episode = task_result["per_episode"]
    return TaskMetrics(
        sum_rewards=[ep["sum_reward"] for ep in per_episode],
        max_rewards=[ep["max_reward"] for ep in per_episode],
        successes=[ep["success"] for ep in per_episode],
        video_paths=task_result.get("video_paths", []),
        trajectory_paths=task_result.get("trajectory_paths", []),
        fiper_rollout_paths=task_result.get("fiper_rollout_paths", []),
    )


def run_one(
    task_group: str,
    task_id: int,
    env,
    *,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
    save_trajectories: bool = True,
    trajectories_dir: Path | None = None,
    save_fiper_rollouts: bool = False,
    fiper_rollouts_dir: Path | None = None,
):
    """
    Run eval_one for a single (task_group, task_id, env).
    Returns (task_group, task_id, task_metrics_dict).
    This function is intentionally module-level to make it easy to test.
    """
    task_videos_dir = None
    if videos_dir is not None:
        task_videos_dir = videos_dir / f"{task_group}_{task_id}"
        task_videos_dir.mkdir(parents=True, exist_ok=True)
    
    task_trajectories_dir = None
    if save_trajectories and trajectories_dir is not None:
        task_trajectories_dir = trajectories_dir / f"{task_group}_{task_id}"
        task_trajectories_dir.mkdir(parents=True, exist_ok=True)

    task_fiper_rollouts_dir = None
    if save_fiper_rollouts and fiper_rollouts_dir is not None:
        task_fiper_rollouts_dir = fiper_rollouts_dir / f"{task_group}_{task_id}"
        task_fiper_rollouts_dir.mkdir(parents=True, exist_ok=True)

    # Call the existing eval_one (assumed to return TaskMetrics-like dict)
    metrics = eval_one(
        env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        save_trajectories=save_trajectories,
        trajectories_dir=task_trajectories_dir,
        save_fiper_rollouts=save_fiper_rollouts,
        fiper_rollouts_dir=task_fiper_rollouts_dir,
    )
    # ensure we always provide these keys to simplify accumulation
    if max_episodes_rendered > 0:
        metrics.setdefault("video_paths", [])
    metrics.setdefault("trajectory_paths", [])
    metrics.setdefault("fiper_rollout_paths", [])
    return task_group, task_id, metrics


def eval_policy_all(
    envs: dict[str, dict[int, gym.vector.VectorEnv]],
    policy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    *,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    max_parallel_tasks: int = 1,
    save_trajectories: bool = True,
    trajectories_dir: Path | None = None,
    save_fiper_rollouts: bool = False,
    fiper_rollouts_dir: Path | None = None,
) -> dict:
    """
    Evaluate a nested `envs` dict: {task_group: {task_id: vec_env}}.
    This implementation flattens tasks, runs them sequentially or via ThreadPoolExecutor,
    accumulates per-group and overall statistics, and returns the same aggregate metrics
    schema as the single-env evaluator (avg_sum_reward / avg_max_reward / pc_success / timings)
    plus per-task infos.
    """
    start_t = time.time()

    # Flatten envs into list of (task_group, task_id, env)
    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]

    # accumulators: track metrics at both per-group level and across all groups
    group_acc: dict[str, dict[str, list]] = defaultdict(lambda: {k: [] for k in ACC_KEYS})
    overall: dict[str, list] = {k: [] for k in ACC_KEYS}
    per_task_infos: list[dict] = []

    # small inline helper to accumulate one task's metrics into accumulators
    def _accumulate_to(group: str, metrics: dict):
        # metrics expected to contain 'sum_rewards', 'max_rewards', 'successes', optionally 'video_paths'
        # but eval_one may store per-episode lists; we assume metrics uses scalars averaged per task as before.
        # To be robust, accept scalars or lists.
        def _append(key, value):
            if value is None:
                return
            if isinstance(value, list):
                group_acc[group][key].extend(value)
                overall[key].extend(value)
            else:
                group_acc[group][key].append(value)
                overall[key].append(value)

        _append("sum_rewards", metrics.get("sum_rewards"))
        _append("max_rewards", metrics.get("max_rewards"))
        _append("successes", metrics.get("successes"))
        # video_paths is list-like
        paths = metrics.get("video_paths", [])
        if paths:
            group_acc[group]["video_paths"].extend(paths)
            overall["video_paths"].extend(paths)
        # trajectory_paths is list-like
        traj_paths = metrics.get("trajectory_paths", [])
        if traj_paths:
            group_acc[group]["trajectory_paths"].extend(traj_paths)
            overall["trajectory_paths"].extend(traj_paths)
        # fiper_rollout_paths is list-like
        fiper_paths = metrics.get("fiper_rollout_paths", [])
        if fiper_paths:
            group_acc[group]["fiper_rollout_paths"].extend(fiper_paths)
            overall["fiper_rollout_paths"].extend(fiper_paths)

    # Choose runner (sequential vs threaded)
    task_runner = partial(
        run_one,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        save_trajectories=save_trajectories,
        trajectories_dir=trajectories_dir,
        save_fiper_rollouts=save_fiper_rollouts,
        fiper_rollouts_dir=fiper_rollouts_dir,
    )

    if max_parallel_tasks <= 1:
        # sequential path (single accumulator path on the main thread)
        # NOTE: keeping a single-threaded accumulator avoids concurrent list appends or locks
        for task_group, task_id, env in tasks:
            tg, tid, metrics = task_runner(task_group, task_id, env)
            _accumulate_to(tg, metrics)
            per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
    else:
        # threaded path: submit all tasks, consume completions on main thread and accumulate there
        with cf.ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
            fut2meta = {}
            for task_group, task_id, env in tasks:
                fut = executor.submit(task_runner, task_group, task_id, env)
                fut2meta[fut] = (task_group, task_id)
            for fut in cf.as_completed(fut2meta):
                tg, tid, metrics = fut.result()
                _accumulate_to(tg, metrics)
                per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})

    # compute aggregated metrics helper (robust to lists/scalars)
    def _agg_from_list(xs):
        if not xs:
            return float("nan")
        arr = np.array(xs, dtype=float)
        return float(np.nanmean(arr))

    # compute per-group aggregates
    groups_aggregated = {}
    for group, acc in group_acc.items():
        groups_aggregated[group] = {
            "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
            "avg_max_reward": _agg_from_list(acc["max_rewards"]),
            "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            "n_episodes": len(acc["sum_rewards"]),
            "video_paths": list(acc["video_paths"]),
            "trajectory_paths": list(acc["trajectory_paths"]),
            "fiper_rollout_paths": list(acc["fiper_rollout_paths"]),
        }

    # overall aggregates
    overall_agg = {
        "avg_sum_reward": _agg_from_list(overall["sum_rewards"]),
        "avg_max_reward": _agg_from_list(overall["max_rewards"]),
        "pc_success": _agg_from_list(overall["successes"]) * 100 if overall["successes"] else float("nan"),
        "n_episodes": len(overall["sum_rewards"]),
        "eval_s": time.time() - start_t,
        "eval_ep_s": (time.time() - start_t) / max(1, len(overall["sum_rewards"])),
        "video_paths": list(overall["video_paths"]),
        "trajectory_paths": list(overall["trajectory_paths"]),
        "fiper_rollout_paths": list(overall["fiper_rollout_paths"]),
    }

    return {
        "per_task": per_task_infos,
        "per_group": groups_aggregated,
        "overall": overall_agg,
    }


def main():
    init_logging()
    register_third_party_plugins()
    eval_main()


if __name__ == "__main__":
    main()
