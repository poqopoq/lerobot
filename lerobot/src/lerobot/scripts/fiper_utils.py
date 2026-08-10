"""
FIPER trajectory collection utilities for LeRobot eval.

Captures per-step data needed by FIPER (Failure Prediction at Runtime for
Generative Robot Policies) and saves it as .pkl files in FIPER's expected format.

Expected .pkl structure per episode:
    {
        "metadata": {"successful": bool, "episode_id": int, ...},
        "rollout": [
            {
                "action":        np.ndarray (action_dim,),
                "action_pred":   np.ndarray (pred_horizon, action_dim),
                "agent_pos":     np.ndarray (state_dim,),
                "obs_embedding": np.ndarray (embed_dim,),
                "rgb":           np.ndarray (H, W, C) uint8,
            },
            ...
        ]
    }
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class FIPERTrajectoryCollector:
    """
    Wraps a policy to capture per-step data required by FIPER:
      - action_pred:   full predicted action chunk (pred_horizon, action_dim)
      - obs_embedding: latent observation embedding from the policy encoder

    Usage:
        collector = FIPERTrajectoryCollector(policy)
        # ... run rollout, calling policy.select_action() each step ...
        # After each step, call:
        action_pred, obs_emb = collector.get_step_extras()
        collector.cleanup()  # remove hooks when done
    """

    def __init__(self, policy: nn.Module):
        self._policy = policy
        self._current_action_pred: torch.Tensor | None = None  # (batch, pred_horizon, action_dim)
        self._current_obs_embedding: torch.Tensor | None = None  # (batch, embed_dim)
        self._hooks: list = []
        self._original_predict_action_chunk = None

        self._patch_predict_action_chunk()
        self._register_obs_embedding_hook()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _patch_predict_action_chunk(self):
        """Monkey-patch the action prediction method to capture action chunks.

        Pi0: select_action() → predict_action_chunk() → captures here.
        SmolVLA: select_action() → _get_action_chunk() → captures here.
        We patch whichever leaf method the policy uses.
        """
        collector = self

        # Prefer patching _get_action_chunk if it exists (covers SmolVLA and similar)
        if hasattr(self._policy, "_get_action_chunk"):
            self._patched_action_attr = "_get_action_chunk"
            original = self._policy._get_action_chunk
            self._original_predict_action_chunk = original

            def patched_get_action(batch, *args, **kwargs):
                chunk = original(batch, *args, **kwargs)
                collector._current_action_pred = chunk.detach().cpu()
                return chunk

            self._policy._get_action_chunk = patched_get_action
        else:
            self._patched_action_attr = "predict_action_chunk"
            original = self._policy.predict_action_chunk
            self._original_predict_action_chunk = original

            def patched(batch, **kwargs):
                chunk = original(batch, **kwargs)
                collector._current_action_pred = chunk.detach().cpu()
                return chunk

            self._policy.predict_action_chunk = patched

    def _register_obs_embedding_hook(self):
        """Try to find the obs encoder and register a hook to capture its output."""
        encoder = self._find_obs_encoder()
        if encoder is None:
            return

        if getattr(self, "_is_pi0", False):
            # Pi0 calls paligemma_with_expert.forward() directly (bypassing __call__),
            # so register_forward_hook won't fire. Monkey-patch .forward() instead.
            original_forward = encoder.forward
            collector = self

            def _patched_forward(*args, **kwargs):
                output = original_forward(*args, **kwargs)
                try:
                    outputs_list, _ = output  # ([prefix, suffix], past_kv)
                    prefix_output = outputs_list[0]  # (batch, seq_len, hidden_dim)
                    suffix_output = outputs_list[1]
                    if (
                        prefix_output is not None
                        and isinstance(prefix_output, torch.Tensor)
                        and suffix_output is None
                    ):
                        # Prefix-only pass — this is the observation encoding step
                        emb = prefix_output.detach().float().mean(dim=1).cpu()
                        collector._current_obs_embedding = emb
                except Exception:
                    pass
                return output

            encoder.forward = _patched_forward
            # Store original so cleanup() can restore it
            self._patched_encoder = encoder
            self._encoder_original_forward = original_forward
        else:
            hook = encoder.register_forward_hook(self._obs_embedding_hook)
            self._hooks.append(hook)

    def _find_obs_encoder(self) -> nn.Module | None:
        """Find the observation encoder for the policy (best-effort, multi-policy support)."""
        p = self._policy

        # Diffusion policy: policy.diffusion.rgb_encoder
        if hasattr(p, "diffusion") and hasattr(p.diffusion, "rgb_encoder"):
            return p.diffusion.rgb_encoder

        # ACT policy: policy.model.encoder
        if hasattr(p, "model") and hasattr(p.model, "encoder"):
            return p.model.encoder

        # Pi0: hook paligemma_with_expert to capture prefix hidden states
        if hasattr(p, "model") and hasattr(p.model, "paligemma_with_expert"):
            self._is_pi0 = True
            return p.model.paligemma_with_expert

        # SmolVLA: hook vlm_with_expert (same prefix/suffix pattern as Pi0)
        if hasattr(p, "model") and hasattr(p.model, "vlm_with_expert"):
            self._is_pi0 = True  # same patching logic applies
            return p.model.vlm_with_expert

        # Other policies: vision_encoder or image_encoder
        for attr in ("vision_encoder", "image_encoder", "visual_encoder", "backbone"):
            if hasattr(p, attr):
                return getattr(p, attr)
            if hasattr(p, "model") and hasattr(p.model, attr):
                return getattr(p.model, attr)

        return None

    def _obs_embedding_hook(self, module, input, output):
        """Forward hook to capture encoder output as obs embedding."""
        # Pi0: paligemma_with_expert returns ([prefix_output, suffix_output], past_key_values)
        # prefix_output is (batch, num_prefix_tokens, hidden_dim) — mean-pool to (batch, hidden_dim)
        if getattr(self, "_is_pi0", False):
            try:
                outputs_list, _ = output  # ([prefix, suffix], past_kv)
                prefix_output = outputs_list[0]  # (batch, seq_len, hidden_dim)
                suffix_output = outputs_list[1]
                if prefix_output is not None and isinstance(prefix_output, torch.Tensor):
                    # Only capture on the prefix-only forward pass (suffix is None)
                    # which is the observation encoding step in sample_actions
                    if suffix_output is None:
                        emb = prefix_output.detach().float().mean(dim=1).cpu()  # (batch, hidden_dim)
                        self._current_obs_embedding = emb
            except Exception:
                pass
            return

        if isinstance(output, torch.Tensor):
            emb = output.detach().cpu()
            # Flatten spatial dims if needed (e.g. (B, C, H, W) → (B, C*H*W))
            if emb.ndim > 2:
                emb = emb.flatten(start_dim=1)
            self._current_obs_embedding = emb
        elif hasattr(output, "last_hidden_state"):
            # Transformer output (e.g. HuggingFace style)
            emb = output.last_hidden_state.detach().cpu()
            # Mean-pool over sequence length → (B, hidden_dim)
            self._current_obs_embedding = emb.mean(dim=1)

    # ------------------------------------------------------------------
    # Per-step data retrieval
    # ------------------------------------------------------------------

    def get_step_extras(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return (action_pred, obs_embedding) captured at the most recent select_action call.

        action_pred:   (batch, pred_horizon, action_dim) or None
        obs_embedding: (batch, embed_dim) or None
        """
        return self._current_action_pred, self._current_obs_embedding

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Remove hooks and restore original action prediction method."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        if self._original_predict_action_chunk is not None:
            attr = getattr(self, "_patched_action_attr", "predict_action_chunk")
            setattr(self._policy, attr, self._original_predict_action_chunk)
        # Restore monkey-patched forward for Pi0
        if hasattr(self, "_patched_encoder") and hasattr(self, "_encoder_original_forward"):
            self._patched_encoder.forward = self._encoder_original_forward


# ---------------------------------------------------------------------------
# PKL saving
# ---------------------------------------------------------------------------


def save_fiper_rollouts_batch(
    rollout_data: dict[str, Any],
    fiper_step_data: list[dict],  # list of length T: [{action_pred, obs_embedding}, ...]
    done_indices: torch.Tensor,
    successes: list[bool],
    start_episode_idx: int,
    output_dir: Path,
) -> list[Path]:
    """
    Save a batch of episodes as FIPER-format .pkl files.

    Args:
        rollout_data: Standard lerobot rollout dict with keys:
            "action"      : (batch, T, action_dim)
            "observation" : nested dict, each tensor (batch, T+1, ...)
        fiper_step_data: List of length T dicts, each with:
            "action_pred"   : (batch, pred_horizon, action_dim) tensor or None
            "obs_embedding" : (batch, embed_dim) tensor or None
        done_indices: (batch,) first done step per env
        successes: list[bool] length batch
        start_episode_idx: episode numbering offset
        output_dir: directory to write .pkl files

    Returns:
        List of saved .pkl paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = rollout_data["action"].shape[0]
    obs = rollout_data.get("observation", {})
    saved_paths = []

    # Determine once which state sub-keys to use for agent_pos
    _state_subkeys = sorted(k for k in obs if k.startswith("observation.state."))

    for batch_idx in range(batch_size):
        ep_len = int(done_indices[batch_idx].item()) + 1
        success = successes[batch_idx]
        episode_idx = start_episode_idx + batch_idx

        rollout_steps = []
        for t in range(ep_len):
            step: dict[str, Any] = {}

            # --- executed action ---
            step["action"] = rollout_data["action"][batch_idx, t].cpu().numpy()

            # --- full predicted action chunk ---
            fiper = fiper_step_data[t] if t < len(fiper_step_data) else {}
            ap = fiper.get("action_pred")
            if ap is not None:
                step["action_pred"] = ap[batch_idx].cpu().numpy()
            else:
                # Fallback: replicate executed action as a single-step "chunk"
                step["action_pred"] = step["action"][None]  # (1, action_dim)

            # --- obs embedding ---
            oe = fiper.get("obs_embedding")
            if oe is not None:
                step["obs_embedding"] = oe[batch_idx].cpu().numpy()
            else:
                step["obs_embedding"] = None

            # --- agent_pos: try flat key, sub-keys, then nested robot_state dict ---
            state = _extract_tensor(obs, "observation.state", batch_idx, t)
            if state is None and _state_subkeys:
                # Pi0/libero-style: state split into sub-keys like
                # observation.state.eef_pos, observation.state.joint_pos, etc.
                parts = [
                    _extract_tensor(obs, k, batch_idx, t).flatten()
                    for k in _state_subkeys
                    if _extract_tensor(obs, k, batch_idx, t) is not None
                ]
                if parts:
                    state = np.concatenate(parts)
            if state is None:
                # Libero: observation.robot_state is a nested dict captured before env_preprocessor
                robot_state_val = obs.get("observation.robot_state")
                if isinstance(robot_state_val, dict):
                    state = _extract_nested_state(robot_state_val, batch_idx, t)
            step["agent_pos"] = state

            # --- rgb from first available image key ---
            rgb = _extract_first_rgb(obs, batch_idx, t)
            step["rgb"] = rgb

            rollout_steps.append(step)

        metadata = {
            "successful": bool(success),
            "episode_id": episode_idx,
        }
        pkl_data = {"metadata": metadata, "rollout": rollout_steps}

        suffix = "success" if success else "fail"
        pkl_path = output_dir / f"episode_{episode_idx:03d}_{suffix}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(pkl_data, f)

        saved_paths.append(pkl_path)

    return saved_paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_tensor(
    obs: dict,
    key: str,
    batch_idx: int,
    t: int,
) -> np.ndarray | None:
    """Extract a single timestep tensor from a nested observation dict."""
    if not obs:
        return None
    # Support both flat key ("observation.state") and nested ("observation" -> "state")
    if key in obs:
        val = obs[key]
    else:
        parts = key.split(".")
        val = obs
        for p in parts:
            if isinstance(val, dict) and p in val:
                val = val[p]
            else:
                return None

    if val is None:
        return None

    try:
        if isinstance(val, torch.Tensor):
            return val[batch_idx, t].cpu().numpy()
        arr = np.asarray(val)
        return arr[batch_idx, t]
    except Exception:
        return None


def _extract_nested_state(
    robot_state: dict,
    batch_idx: int,
    t: int,
) -> np.ndarray | None:
    """Flatten a nested robot_state dict (libero format) into a 1D agent_pos vector.

    Expected structure (from libero.py):
        {"eef": {"pos": T, "quat": T, "mat": T},
         "gripper": {"qpos": T, "qvel": T},
         "joints": {"pos": T, "vel": T}}
    where T is a tensor of shape (batch, seq, ...).
    """
    parts = []

    def _collect(d: dict) -> None:
        for v in d.values():
            if isinstance(v, dict):
                _collect(v)
            elif isinstance(v, torch.Tensor):
                try:
                    arr = v[batch_idx, t].cpu().numpy().flatten()
                    parts.append(arr)
                except Exception:
                    pass
            elif v is not None:
                try:
                    arr = np.asarray(v)[batch_idx, t].flatten()
                    parts.append(arr)
                except Exception:
                    pass

    _collect(robot_state)
    return np.concatenate(parts) if parts else None


def _extract_first_rgb(
    obs: dict,
    batch_idx: int,
    t: int,
) -> np.ndarray | None:
    """Extract the first RGB image from observations."""
    if not obs:
        return None

    # Priority: look for keys ending in _rgb or containing 'image'
    def _search(d: dict, depth: int = 0):
        if depth > 3:
            return None
        for k, v in d.items():
            if isinstance(v, dict):
                result = _search(v, depth + 1)
                if result is not None:
                    return result
            elif isinstance(v, (torch.Tensor, np.ndarray)):
                key_lower = k.lower()
                if any(x in key_lower for x in ("rgb", "image", "img", "camera")):
                    try:
                        if isinstance(v, torch.Tensor):
                            arr = v[batch_idx, t].cpu().numpy()
                        else:
                            arr = np.asarray(v)[batch_idx, t]
                        # Ensure uint8 (H, W, C) — handle (C, H, W) if needed
                        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                            arr = np.transpose(arr, (1, 2, 0))
                        if arr.dtype != np.uint8:
                            if arr.max() <= 1.0:
                                arr = (arr * 255).astype(np.uint8)
                            else:
                                arr = arr.astype(np.uint8)
                        return arr
                    except Exception:
                        continue
        return None

    return _search(obs)
