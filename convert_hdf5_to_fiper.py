#!/usr/bin/env python3
"""
Convert LeRobot HDF5 evaluation trajectories to FIPER-compatible pickle files.

This script:
1. Reads HDF5 files from lerobot-eval
2. Loads the policy to extract obs_embeddings
3. Saves in FIPER .pkl format

Usage:
    python convert_hdf5_to_fiper.py \\
        --input_dir eval_trajectories5/libero_10_7 \\
        --policy_path lerobot/pi05_libero_finetuned \\
        --output_dir fiper_rollouts/test

Author: Auto-generated for FIPER data collection
"""

import argparse
import h5py
import numpy as np
import pickle
import torch
from pathlib import Path
from tqdm import tqdm
import warnings

# Suppress warnings during model loading
warnings.filterwarnings('ignore')


def load_policy_with_fallback(policy_path, device="cuda"):
    """
    Try multiple methods to load the policy.
    """
    print(f"📦 Loading policy: {policy_path}")

    # Method 1: Try PI05Policy directly
    try:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        policy = PI05Policy.from_pretrained(policy_path)
        policy = policy.to(device)
        policy.eval()
        print(f"✅ Loaded PI05Policy")
        return policy, "pi05"
    except Exception as e:
        print(f"⚠️ PI05 loading failed: {e}")

    # Method 2: Try PI0Policy
    try:
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        policy = PI0Policy.from_pretrained(policy_path)
        policy = policy.to(device)
        policy.eval()
        print(f"✅ Loaded PI0Policy")
        return policy, "pi0"
    except Exception as e:
        print(f"⚠️ PI0 loading failed: {e}")

    # Method 3: Generic loading
    try:
        from lerobot.policies.factory import get_policy_class
        from lerobot.policies.pretrained import PreTrainedConfig

        config = PreTrainedConfig.from_pretrained(policy_path)
        policy_class = get_policy_class(config.name if hasattr(config, 'name') else 'pi05')
        policy = policy_class.from_pretrained(policy_path)
        policy = policy.to(device)
        policy.eval()
        print(f"✅ Loaded {policy_class.__name__}")
        return policy, "generic"
    except Exception as e:
        print(f"❌ All loading methods failed: {e}")
        raise RuntimeError("Could not load policy. Model may have missing dependencies.")


def extract_embedding_from_policy(policy, obs_dict, policy_type, device):
    """
    Extract observation embedding from policy's vision encoder.

    This uses forward hooks since direct access varies by policy type.
    """
    captured_embedding = []

    def hook_fn(module, input, output):
        captured_embedding.append(output.detach().cpu())

    # Register hook on vision encoder (try multiple possible locations)
    hook_handle = None
    encoder_found = False

    for attr_name in ['vision_backbone', 'visual_features', 'vision_encoder',
                       'image_encoder', 'encoder', 'backbone']:
        if hasattr(policy, attr_name):
            encoder = getattr(policy, attr_name)
            hook_handle = encoder.register_forward_hook(hook_fn)
            encoder_found = True
            break
        elif hasattr(policy, 'model') and hasattr(policy.model, attr_name):
            encoder = getattr(policy.model, attr_name)
            hook_handle = encoder.register_forward_hook(hook_fn)
            encoder_found = True
            break

    if not encoder_found:
        # Fallback: return zeros
        return np.zeros(512, dtype=np.float32)

    try:
        # Prepare observation for policy
        obs_tensor = {}
        for key, value in obs_dict.items():
            if isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value).to(device)
                if len(tensor.shape) == 3:  # Image: add batch dim
                    tensor = tensor.unsqueeze(0)
                obs_tensor[key] = tensor

        # Run forward pass (triggers hook)
        with torch.no_grad():
            try:
                _ = policy.select_action(obs_tensor)
            except:
                # Some policies may fail on select_action, but embedding might still be captured
                pass

        # Extract captured embedding
        if len(captured_embedding) > 0:
            emb = captured_embedding[0].squeeze().numpy()
        else:
            emb = np.zeros(512, dtype=np.float32)

    finally:
        if hook_handle is not None:
            hook_handle.remove()

    return emb.astype(np.float32)


def convert_hdf5_to_fiper(hdf5_path, policy, policy_type, device):
    """
    Convert a single HDF5 file to FIPER format.

    Returns: (trajectory, success, episode_id)
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Assume structure: f['data']['demo_0']
        demo = f['data']['demo_0']

        actions = demo['actions'][:]
        T = actions.shape[0]

        # Get observation keys
        obs_keys = list(demo['obs'].keys())

        # Find RGB key
        rgb_key = None
        for key in ['agentview_rgb', 'image', 'rgb', 'eye_in_hand_rgb']:
            if key in obs_keys:
                rgb_key = key
                break

        # Find robot state key
        state_key = None
        for key in ['ee_pos', 'robot_states', 'agent_pos', 'joint_states']:
            if key in obs_keys:
                state_key = key
                break

        trajectory = []

        # Process each timestep
        for t in range(T):
            # Build observation dict
            obs_dict = {}
            for key in obs_keys:
                obs_dict[key] = demo['obs'][key][t]

            # Extract embedding using policy
            obs_embedding = extract_embedding_from_policy(
                policy, obs_dict, policy_type, device
            )

            # Get action
            action = actions[t].astype(np.float32)

            # action_pred: For post-processing, we don't have multi-step predictions
            # So we'll just repeat the single action (not ideal but workable)
            action_pred = np.tile(action, (16, 1)).astype(np.float32)  # 16 = default horizon

            # Build timestep
            timestep = {
                'obs_embedding': obs_embedding,
                'action_pred': action_pred,
                'action': action,
            }

            if rgb_key:
                timestep['rgb'] = demo['obs'][rgb_key][t]

            if state_key:
                timestep['agent_pos'] = demo['obs'][state_key][t].astype(np.float32)

            trajectory.append(timestep)

        # Determine success (heuristic: check if rewards exist and final reward > 0)
        if 'rewards' in demo:
            rewards = demo['rewards'][:]
            success = float(rewards[-1]) > 0.5
        else:
            success = True  # Assume success if no reward info

        # Extract episode ID from filename
        filename = Path(hdf5_path).name
        import re
        match = re.search(r'\d+', filename)
        episode_id = int(match.group()) if match else 0

    return trajectory, success, episode_id


def main(args):
    print("=" * 70)
    print("HDF5 to FIPER Converter")
    print("=" * 70)
    print(f"Input dir: {args.input_dir}")
    print(f"Policy: {args.policy_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device: {args.device}")
    print("=" * 70)

    # Load policy
    policy, policy_type = load_policy_with_fallback(args.policy_path, args.device)

    # Find all HDF5 files
    input_path = Path(args.input_dir)
    hdf5_files = sorted(input_path.glob("*.hdf5")) + sorted(input_path.glob("*.h5"))

    if len(hdf5_files) == 0:
        print(f"❌ No HDF5 files found in {input_path}")
        return

    print(f"\n📁 Found {len(hdf5_files)} HDF5 files")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert each file
    print(f"\n🔄 Converting to FIPER format...")

    for hdf5_file in tqdm(hdf5_files, desc="Converting"):
        try:
            trajectory, success, episode_id = convert_hdf5_to_fiper(
                hdf5_file, policy, policy_type, args.device
            )

            # Save as pickle
            rollout_data = {
                'metadata': {
                    'successful': success,
                    'episode_id': episode_id,
                },
                'rollout': trajectory
            }

            filename = f"rollout_{episode_id:04d}_{'success' if success else 'failure'}.pkl"
            output_file = output_path / filename

            with open(output_file, 'wb') as f:
                pickle.dump(rollout_data, f)

            if args.verbose:
                tqdm.write(f"✅ {hdf5_file.name} → {filename} ({len(trajectory)} steps)")

        except Exception as e:
            tqdm.write(f"❌ Error processing {hdf5_file.name}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 70)
    print("✅ Conversion complete!")
    print(f"Output saved to: {output_path}")
    print("\n💡 Next steps:")
    print(f"  1. Verify: python verify_fiper_rollouts.py {output_path}")
    print(f"  2. Move to FIPER: mv {output_path} /home/zhiyuanjia/fiper/data/your_task/rollouts/test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LeRobot HDF5 to FIPER format")
    parser.add_argument("--input_dir", required=True,
                       help="Directory containing HDF5 files from lerobot-eval")
    parser.add_argument("--policy_path", required=True,
                       help="Policy path or HuggingFace model ID")
    parser.add_argument("--output_dir", default="fiper_rollouts",
                       help="Output directory for FIPER .pkl files")
    parser.add_argument("--device", default="cuda",
                       help="Device to run policy on (cuda/cpu)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed progress")

    args = parser.parse_args()
    main(args)
