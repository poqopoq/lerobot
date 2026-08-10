#!/usr/bin/env python3
"""
Verify that collected rollouts are compatible with FIPER.
"""

import pickle
import numpy as np
from pathlib import Path
import argparse


def verify_rollout_file(filepath):
    """Verify a single rollout file"""
    errors = []
    warnings = []

    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        return [f"Cannot load file: {e}"], []

    # Check structure
    if isinstance(data, dict):
        if "rollout" not in data:
            errors.append("Missing 'rollout' key in dictionary")
        else:
            rollout = data["rollout"]

        if "metadata" in data:
            metadata = data["metadata"]
            if "successful" not in metadata:
                warnings.append("Metadata missing 'successful' field")
        else:
            warnings.append("Missing 'metadata' key")
    elif isinstance(data, list):
        rollout = data
        warnings.append("Using list format (metadata will be from filename)")
    else:
        errors.append(f"Unknown format: {type(data)}")
        return errors, warnings

    # Check rollout is a list
    if not isinstance(rollout, list):
        errors.append(f"Rollout should be list, got {type(rollout)}")
        return errors, warnings

    if len(rollout) == 0:
        errors.append("Empty rollout")
        return errors, warnings

    # Check first timestep
    timestep = rollout[0]
    if not isinstance(timestep, dict):
        errors.append(f"Timestep should be dict, got {type(timestep)}")
        return errors, warnings

    # Check required keys
    required_keys = ["obs_embedding", "action_pred"]
    for key in required_keys:
        if key not in timestep:
            errors.append(f"Missing required key: '{key}'")

    # Check optional keys
    optional_keys = ["action", "rgb", "agent_pos"]
    for key in optional_keys:
        if key not in timestep:
            warnings.append(f"Missing optional key: '{key}'")

    # Check shapes
    if "obs_embedding" in timestep:
        emb = timestep["obs_embedding"]
        if not isinstance(emb, np.ndarray):
            errors.append(f"obs_embedding should be numpy array, got {type(emb)}")
        elif len(emb.shape) != 1:
            errors.append(f"obs_embedding should be 1D, got shape {emb.shape}")
        elif emb.shape[0] == 0:
            errors.append("obs_embedding is empty")
        elif np.all(emb == 0):
            warnings.append(f"obs_embedding is all zeros (shape: {emb.shape})")

    if "action_pred" in timestep:
        action = timestep["action_pred"]
        if not isinstance(action, np.ndarray):
            errors.append(f"action_pred should be numpy array, got {type(action)}")
        elif len(action.shape) != 2:
            warnings.append(f"action_pred usually 2D [horizon, action_dim], got shape {action.shape}")

    return errors, warnings


def verify_rollout_directory(directory):
    """Verify all rollouts in a directory"""
    directory = Path(directory)

    if not directory.exists():
        print(f"❌ Directory does not exist: {directory}")
        return

    pkl_files = list(directory.glob("*.pkl"))

    if len(pkl_files) == 0:
        print(f"⚠️  No .pkl files found in {directory}")
        return

    print(f"\n📁 Verifying {len(pkl_files)} files in {directory}")
    print("=" * 70)

    total_errors = 0
    total_warnings = 0
    problematic_files = []

    for pkl_file in pkl_files:
        errors, warnings = verify_rollout_file(pkl_file)

        if errors:
            total_errors += len(errors)
            problematic_files.append(pkl_file.name)
            print(f"\n❌ {pkl_file.name}")
            for error in errors:
                print(f"   ERROR: {error}")
            for warning in warnings:
                print(f"   WARNING: {warning}")
        elif warnings:
            total_warnings += len(warnings)
            print(f"\n⚠️  {pkl_file.name}")
            for warning in warnings:
                print(f"   WARNING: {warning}")

    print("\n" + "=" * 70)
    if total_errors == 0 and total_warnings == 0:
        print(f"✅ All {len(pkl_files)} files are valid!")
    elif total_errors == 0:
        print(f"✅ All files valid with {total_warnings} warnings")
    else:
        print(f"❌ Found {total_errors} errors and {total_warnings} warnings")
        print(f"   Problematic files: {len(problematic_files)}/{len(pkl_files)}")

    # Show sample data
    if len(pkl_files) > 0:
        print(f"\n📊 Sample data from {pkl_files[0].name}:")
        print("-" * 70)
        with open(pkl_files[0], 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict) and "rollout" in data:
            rollout = data["rollout"]
            print(f"Metadata: {data.get('metadata', {})}")
        else:
            rollout = data

        if len(rollout) > 0:
            timestep = rollout[0]
            print(f"Timestep keys: {list(timestep.keys())}")
            for key, value in timestep.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"  {key}: {type(value)}")
            print(f"Episode length: {len(rollout)} timesteps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify FIPER rollout format")
    parser.add_argument("directory", help="Directory containing .pkl rollout files")

    args = parser.parse_args()

    verify_rollout_directory(args.directory)
