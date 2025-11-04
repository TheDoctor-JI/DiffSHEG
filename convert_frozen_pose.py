#!/usr/bin/env python3
"""
Convert Frozen Skeleton Pose to Normalized Custom Neutral Positions

This script converts a frozen skeleton pose (exported from the test mode GUI)
into normalized custom neutral positions that can be used in the DiffSHEG wrapper.

CONVERSION PROCESS:
===================

1. Input: frozen_skeleton_*.yaml
   - Contains denormalized Euler angles in degrees (as displayed in Three.js)
   - These are the actual joint rotations from the frozen pose

2. Normalization: 
   - Uses the same bvh_mean.npy and bvh_std.npy as the training data
   - Formula: normalized = (denormalized - mean) / std
   
3. Output: custom_neutral_positions_*.yaml
   - Contains normalized values ready for use in embodiment_manager_config.yaml
   - Can be copied directly into the co_speech_gestures section

USAGE:
======

python convert_frozen_pose.py \\
    --input frozen_skeleton_1234567890.yaml \\
    --cache_name beat_4english_15_141 \\
    --output custom_neutral_positions.yaml

Then copy the output YAML content to embodiment_manager_config.yaml:

co_speech_gestures:
  joint_mask:
    - RightArm
    - RightForeArm
    # ... etc
  custom_neutral_positions:
    RightArm: [normalized_x, normalized_y, normalized_z]
    RightForeArm: [normalized_x, normalized_y, normalized_z]
    # ... etc (paste from output)
"""

import os
import sys
import yaml
import numpy as np
import argparse
from pathlib import Path

# BEAT skeleton joint order - must match diffsheg_realtime_wrapper.py and read_frame_val.py
BEAT_GESTURE_JOINT_ORDER = [
    'Spine',          # indices 0-2
    'Neck',           # indices 3-5
    'Neck1',          # indices 6-8
    'RightShoulder',  # indices 9-11
    'RightArm',       # indices 12-14
    'RightForeArm',   # indices 15-17
    'RightHand',      # indices 18-20
    'RightHandMiddle1',  # indices 21-23
    'RightHandMiddle2',  # indices 24-26
    'RightHandMiddle3',  # indices 27-29
    'RightHandRing',     # indices 30-32
    'RightHandRing1',    # indices 33-35
    'RightHandRing2',    # indices 36-38
    'RightHandRing3',    # indices 39-41
    'RightHandPinky',    # indices 42-44
    'RightHandPinky1',   # indices 45-47
    'RightHandPinky2',   # indices 48-50
    'RightHandPinky3',   # indices 51-53
    'RightHandIndex',    # indices 54-56
    'RightHandIndex1',   # indices 57-59
    'RightHandIndex2',   # indices 60-62
    'RightHandIndex3',   # indices 63-65
    'RightHandThumb1',   # indices 66-68
    'RightHandThumb2',   # indices 69-71
    'RightHandThumb3',   # indices 72-74
    'LeftShoulder',      # indices 75-77
    'LeftArm',           # indices 78-80
    'LeftForeArm',       # indices 81-83
    'LeftHand',          # indices 84-86
    'LeftHandMiddle1',   # indices 87-89
    'LeftHandMiddle2',   # indices 90-92
    'LeftHandMiddle3',   # indices 93-95
    'LeftHandRing',      # indices 96-98
    'LeftHandRing1',     # indices 99-101
    'LeftHandRing2',     # indices 102-104
    'LeftHandRing3',     # indices 105-107
    'LeftHandPinky',     # indices 108-110
    'LeftHandPinky1',    # indices 111-113
    'LeftHandPinky2',    # indices 114-116
    'LeftHandPinky3',    # indices 117-119
    'LeftHandIndex',     # indices 120-122
    'LeftHandIndex1',    # indices 123-125
    'LeftHandIndex2',    # indices 126-128
    'LeftHandIndex3',    # indices 129-131
    'LeftHandThumb1',    # indices 132-134
    'LeftHandThumb2',    # indices 135-137
    'LeftHandThumb3',    # indices 138-140
]


def load_frozen_pose(yaml_path: str) -> dict:
    """
    Load frozen skeleton pose from YAML file.
    
    Args:
        yaml_path: Path to frozen_skeleton_*.yaml
        
    Returns:
        Dictionary mapping joint_name -> [x, y, z] in degrees (denormalized)
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if 'frozen_joint_rotations' not in data:
        raise ValueError(
            f"Invalid frozen pose file: missing 'frozen_joint_rotations' key. "
            f"Expected format: frozen_joint_rotations:\\n  JointName: [x, y, z]"
        )
    
    return data['frozen_joint_rotations']


def load_normalization_stats(cache_name: str, script_dir: str):
    """
    Load normalization statistics (mean and std) for Euler angles.
    
    Args:
        cache_name: Cache name (e.g., 'beat_4english_15_141')
        script_dir: Directory containing this script
        
    Returns:
        Tuple of (mean_array, std_array) as numpy arrays of shape (141,)
    """
    # Construct paths matching co_speech_gesture_server.py
    stats_dir = os.path.join(script_dir, "data", "BEAT", "beat_cache", cache_name, "train")
    
    mean_euler_path = os.path.join(stats_dir, "bvh_rot", "bvh_mean.npy")
    std_euler_path = os.path.join(stats_dir, "bvh_rot", "bvh_std.npy")
    
    if not os.path.exists(mean_euler_path):
        raise FileNotFoundError(
            f"Euler mean file not found: {mean_euler_path}\n"
            f"Make sure cache_name '{cache_name}' is correct and data exists."
        )
    
    if not os.path.exists(std_euler_path):
        raise FileNotFoundError(
            f"Euler std file not found: {std_euler_path}\n"
            f"Make sure cache_name '{cache_name}' is correct and data exists."
        )
    
    mean_euler = np.load(mean_euler_path)
    std_euler = np.load(std_euler_path)
    
    # Validate dimensions
    if mean_euler.shape[0] != 141:
        raise ValueError(
            f"Invalid mean_euler shape: {mean_euler.shape}, expected (141,). "
            f"This should contain 47 joints × 3 = 141 dimensions."
        )
    
    if std_euler.shape[0] != 141:
        raise ValueError(
            f"Invalid std_euler shape: {std_euler.shape}, expected (141,). "
            f"This should contain 47 joints × 3 = 141 dimensions."
        )
    
    return mean_euler, std_euler


def normalize_euler_angles(denormalized_angles: dict, mean_pose: np.ndarray, std_pose: np.ndarray) -> dict:
    """
    Convert denormalized Euler angles to normalized space.
    
    This is the INVERSE of the denormalization process:
    - Denormalization (in waypoint_post_proc): denorm = norm * std + mean
    - Normalization (this function): norm = (denorm - mean) / std
    
    Args:
        denormalized_angles: Dict of joint_name -> [x, y, z] in degrees (from frozen pose)
        mean_pose: numpy array of shape (141,) - Euler mean from bvh_mean.npy
        std_pose: numpy array of shape (141,) - Euler std from bvh_std.npy
    
    Returns:
        Dict of joint_name -> [x_norm, y_norm, z_norm] in normalized space
    """
    normalized = {}
    
    for joint_name, angles in denormalized_angles.items():
        # Validate joint name
        if joint_name not in BEAT_GESTURE_JOINT_ORDER:
            print(f"⚠️  Warning: Joint '{joint_name}' not in BEAT_GESTURE_JOINT_ORDER, skipping")
            continue
        
        # Find this joint's index in the 141-dimensional array
        joint_idx = BEAT_GESTURE_JOINT_ORDER.index(joint_name)
        start_idx = joint_idx * 3
        
        # Validate angle format
        if not isinstance(angles, (list, tuple)) or len(angles) != 3:
            print(f"⚠️  Warning: Invalid angles for '{joint_name}': {angles}, skipping")
            continue
        
        # Extract denormalized values (in degrees)
        denorm = np.array(angles, dtype=np.float32)
        
        # Get corresponding mean/std for this joint (3 channels: X, Y, Z)
        mean_vals = mean_pose[start_idx:start_idx+3]
        std_vals = std_pose[start_idx:start_idx+3]
        
        # Normalize: (denorm - mean) / std
        norm_vals = (denorm - mean_vals) / std_vals
        
        normalized[joint_name] = norm_vals.tolist()
    
    return normalized


def save_custom_neutral_positions(normalized_angles: dict, output_path: str):
    """
    Save normalized angles as custom_neutral_positions YAML.
    
    Args:
        normalized_angles: Dict of joint_name -> [x_norm, y_norm, z_norm]
        output_path: Output YAML file path
    """
    # Create output content
    yaml_content = {
        'custom_neutral_positions': {}
    }
    
    # Add joints in BEAT order
    for joint_name in BEAT_GESTURE_JOINT_ORDER:
        if joint_name in normalized_angles:
            # Format with 6 decimal places
            angles = normalized_angles[joint_name]
            yaml_content['custom_neutral_positions'][joint_name] = [
                round(angles[0], 6),
                round(angles[1], 6),
                round(angles[2], 6)
            ]
    
    # Save to file
    with open(output_path, 'w') as f:
        # Write header comments
        f.write('# Custom Neutral Positions for DiffSHEG Wrapper\n')
        f.write('# Generated from frozen skeleton pose\n')
        f.write('# These values are NORMALIZED and ready for use in embodiment_manager_config.yaml\n')
        f.write('#\n')
        f.write('# USAGE:\n')
        f.write('# Copy the custom_neutral_positions section to your embodiment_manager_config.yaml:\n')
        f.write('#\n')
        f.write('# co_speech_gestures:\n')
        f.write('#   joint_mask:\n')
        f.write('#     - RightArm\n')
        f.write('#     - LeftArm\n')
        f.write('#     # ... (list joints you want to mask)\n')
        f.write('#   custom_neutral_positions:\n')
        f.write('#     # Paste the content below here\n')
        f.write('\n')
        
        # Write YAML data
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ Saved custom neutral positions to: {output_path}")


def print_summary(denormalized: dict, normalized: dict):
    """Print a summary comparison of denormalized vs normalized values."""
    print("\n" + "="*80)
    print("CONVERSION SUMMARY")
    print("="*80)
    print(f"Total joints converted: {len(normalized)}")
    print("\nSample conversions (first 5 joints):")
    print("-"*80)
    print(f"{'Joint Name':<20} {'Denormalized (degrees)':<30} {'Normalized':<30}")
    print("-"*80)
    
    count = 0
    for joint_name in BEAT_GESTURE_JOINT_ORDER:
        if joint_name in normalized and count < 5:
            denorm = denormalized[joint_name]
            norm = normalized[joint_name]
            
            denorm_str = f"[{denorm[0]:7.2f}, {denorm[1]:7.2f}, {denorm[2]:7.2f}]"
            norm_str = f"[{norm[0]:7.4f}, {norm[1]:7.4f}, {norm[2]:7.4f}]"
            
            print(f"{joint_name:<20} {denorm_str:<30} {norm_str:<30}")
            count += 1
    
    print("-"*80)


def main():
    parser = argparse.ArgumentParser(
        description='Convert frozen skeleton pose to normalized custom neutral positions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input frozen skeleton YAML file (frozen_skeleton_*.yaml)'
    )
    
    parser.add_argument(
        '--cache_name',
        type=str,
        default='beat_4english_15_141',
        help='Cache name for loading normalization statistics (default: beat_4english_15_141)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output YAML file (default: custom_neutral_positions_<timestamp>.yaml)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"✗ Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Set default output name
    if args.output is None:
        import time
        timestamp = int(time.time())
        args.output = f"custom_neutral_positions_{timestamp}.yaml"
    
    print("="*80)
    print("FROZEN POSE TO CUSTOM NEUTRAL POSITIONS CONVERTER")
    print("="*80)
    print(f"Input file:  {args.input}")
    print(f"Cache name:  {args.cache_name}")
    print(f"Output file: {args.output}")
    print("="*80)
    
    try:
        # Step 1: Load frozen pose
        print("\n[1/4] Loading frozen skeleton pose...")
        denormalized_angles = load_frozen_pose(args.input)
        print(f"      ✓ Loaded {len(denormalized_angles)} joints")
        
        # Step 2: Load normalization statistics
        print("\n[2/4] Loading normalization statistics...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mean_euler, std_euler = load_normalization_stats(args.cache_name, script_dir)
        print(f"      ✓ Loaded mean and std from cache '{args.cache_name}'")
        print(f"      - Mean shape: {mean_euler.shape}")
        print(f"      - Std shape:  {std_euler.shape}")
        
        # Step 3: Normalize angles
        print("\n[3/4] Converting to normalized space...")
        normalized_angles = normalize_euler_angles(denormalized_angles, mean_euler, std_euler)
        print(f"      ✓ Normalized {len(normalized_angles)} joints")
        
        # Step 4: Save output
        print("\n[4/4] Saving custom neutral positions...")
        save_custom_neutral_positions(normalized_angles, args.output)
        
        # Print summary
        print_summary(denormalized_angles, normalized_angles)
        
        # Print usage instructions
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. Open your embodiment_manager_config.yaml")
        print("\n2. Add the custom_neutral_positions to the co_speech_gestures section:")
        print("\n   co_speech_gestures:")
        print("     joint_mask:")
        print("       - RightArm")
        print("       - LeftArm")
        print("       # ... (list the joints you want to mask)")
        print("\n     custom_neutral_positions:")
        print(f"       # Copy content from: {args.output}")
        print("\n3. Restart the co_speech_gesture_server")
        print("\n4. Masked joints will now use your custom neutral positions!")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
