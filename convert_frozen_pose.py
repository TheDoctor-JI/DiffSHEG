#!/usr/bin/env python3
"""
Convert Frozen Skeleton Pose to Normalized Custom Neutral Positions

This script converts a frozen skeleton pose (exported from the test mode GUI)
into normalized custom neutral positions that can be used in the DiffSHEG wrapper.

CRITICAL INFORMATION ABOUT ANGULAR FORMAT:
===========================================

The model uses axis_angle=True (see opt.txt), which means:
1. Model outputs: NORMALIZED axis-angle representation
2. Wrapper applies joint mask: Uses custom_neutral_positions in NORMALIZED axis-angle space
3. Server waypoint_post_proc: Converts NORMALIZED axis-angle → DENORMALIZED Euler degrees
4. Frontend displays: DENORMALIZED Euler degrees
5. Frozen pose export: DENORMALIZED Euler degrees
6. This script: Converts DENORMALIZED Euler degrees → NORMALIZED axis-angle

CONVERSION PROCESS:
===================

1. Input: frozen_skeleton_*.yaml
   - Contains denormalized Euler angles in degrees (as displayed in Three.js)
   - These are the actual joint rotations from the frozen pose

2. Conversion Steps (REVERSE of waypoint_post_proc):
   a. Start with: Denormalized Euler degrees (from frozen pose)
   b. Convert to: Euler radians
   c. Convert to: axis-angle representation
   d. Normalize: (axis_angle - mean_axis) / std_axis
   
   This produces NORMALIZED axis-angle values that can be used directly
   in the wrapper's joint masking, matching the model output format
   
3. Output: custom_neutral_positions_*.yaml
   - Contains normalized values ready for use in embodiment_manager_config.yaml
   - Can be copied directly into the co_speech_gestures section

USAGE:
======

python convert_frozen_pose.py \\
    --input frozen_skeleton_1234567890.yaml \\
    --output custom_neutral_positions.yaml

NOTE: The cache name is HARDCODED to 'beat_4english_15_141' to match the server configuration.
This ensures the normalization statistics used here EXACTLY match what the co-speech gesture
server uses for denormalization.

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
import torch

# Add DiffSHEG to path for rotation conversion utilities
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

try:
    import datasets.rotation_converter as rot_cvt
except ImportError as e:
    print(f"ERROR: Failed to import rotation_converter from DiffSHEG datasets: {e}")
    print("Make sure you're running this script from the DiffSHEG directory.")
    sys.exit(1)

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
    Load normalization statistics for axis-angle representation.
    
    CRITICAL: This function MUST use the exact same paths and statistics as
    co_speech_gesture_server.py to ensure the conversion is correct.
    
    Since the model uses axis_angle=True, we need axis-angle statistics
    to convert frozen poses (Euler degrees) back to normalized axis-angle.
    
    Args:
        cache_name: Cache name (HARDCODED to 'beat_4english_15_141' in main())
        script_dir: Directory containing this script
        
    Returns:
        Tuple of (mean_axis, std_axis) as numpy arrays of shape (141,)
    """
    # Construct paths matching co_speech_gesture_server.py EXACTLY
    # Server path: os.path.join(script_dir, "DiffSHEG", "data", "BEAT", "beat_cache", opt.beat_cache_name, "train")
    # This script is IN DiffSHEG/, so we don't need the "DiffSHEG" part
    stats_dir = os.path.join(script_dir, "data", "BEAT", "beat_cache", cache_name, "train")
    
    mean_axis_path = os.path.join(stats_dir, "axis_angle_mean.npy")
    std_axis_path = os.path.join(stats_dir, "axis_angle_std.npy")
    
    # Print paths for verification
    print(f"      Loading AXIS-ANGLE statistics (model uses axis_angle=True):")
    print(f"      - Mean: {mean_axis_path}")
    print(f"      - Std:  {std_axis_path}")
    
    if not os.path.exists(mean_axis_path):
        raise FileNotFoundError(
            f"Axis-angle mean file not found: {mean_axis_path}\n"
            f"Make sure cache_name '{cache_name}' is correct and data exists."
        )
    
    if not os.path.exists(std_axis_path):
        raise FileNotFoundError(
            f"Axis-angle std file not found: {std_axis_path}\n"
            f"Make sure cache_name '{cache_name}' is correct and data exists."
        )
    
    mean_axis = np.load(mean_axis_path)
    std_axis = np.load(std_axis_path)
    
    # Validate dimensions
    if mean_axis.shape[0] != 141:
        raise ValueError(
            f"Invalid mean_axis shape: {mean_axis.shape}, expected (141,). "
            f"This should contain 47 joints × 3 = 141 dimensions."
        )
    
    if std_axis.shape[0] != 141:
        raise ValueError(
            f"Invalid std_axis shape: {std_axis.shape}, expected (141,). "
            f"This should contain 47 joints × 3 = 141 dimensions."
        )
    
    return mean_axis, std_axis


def convert_euler_to_normalized_axis_angle(denormalized_angles: dict, mean_axis: np.ndarray, std_axis: np.ndarray) -> dict:
    """
    Convert denormalized Euler angles (degrees) to normalized axis-angle representation.
    
    This is the COMPLETE INVERSE of waypoint_post_proc() with axis_angle=True:
    
    Server waypoint_post_proc flow:
    1. Input: NORMALIZED axis-angle (from model output)
    2. Denormalize: denorm_axis = norm_axis * std_axis + mean_axis
    3. Convert: axis-angle → Euler (radians) → Euler (degrees)
    4. Output: DENORMALIZED Euler degrees (to frontend)
    
    This function does the REVERSE:
    1. Input: DENORMALIZED Euler degrees (from frozen pose)
    2. Convert: Euler (degrees) → Euler (radians) → axis-angle
    3. Normalize: norm_axis = (axis_angle - mean_axis) / std_axis
    4. Output: NORMALIZED axis-angle (for custom_neutral_positions)
    
    Args:
        denormalized_angles: Dict of joint_name -> [x, y, z] in degrees (from frozen pose)
        mean_axis: numpy array of shape (141,) - axis-angle mean from axis_angle_mean.npy
        std_axis: numpy array of shape (141,) - axis-angle std from axis_angle_std.npy
    
    Returns:
        Dict of joint_name -> [x_norm, y_norm, z_norm] in normalized axis-angle space
    """
    normalized = {}
    
    # Collect all Euler angles in order for batch conversion
    euler_degrees_full = np.zeros(141, dtype=np.float32)
    
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
        
        # Store in full array
        euler_degrees_full[start_idx:start_idx+3] = angles
    
    # Convert to radians
    euler_radians = euler_degrees_full * (np.pi / 180.0)
    
    # Reshape for rotation conversion: (141,) → (47, 3)
    euler_radians_reshaped = euler_radians.reshape(47, 3)
    
    # Convert Euler (radians) → axis-angle using DiffSHEG's rotation utilities
    # This is the INVERSE of what waypoint_post_proc does
    # Note: BEAT uses 'XYZ' convention (same as used in axis_angle_to_euler_angles)
    euler_tensor = torch.from_numpy(euler_radians_reshaped).float()
    axis_angle_tensor = rot_cvt.euler_angles_to_axis_angle(euler_tensor, 'XYZ')  # (47, 3)
    axis_angle = axis_angle_tensor.numpy().reshape(-1)  # (141,)
    
    # Normalize: (axis_angle - mean) / std
    normalized_axis_angle = (axis_angle - mean_axis) / std_axis
    
    # Extract per-joint values
    for joint_name in denormalized_angles.keys():
        if joint_name not in BEAT_GESTURE_JOINT_ORDER:
            continue
            
        joint_idx = BEAT_GESTURE_JOINT_ORDER.index(joint_name)
        start_idx = joint_idx * 3
        
        norm_vals = normalized_axis_angle[start_idx:start_idx+3]
        normalized[joint_name] = norm_vals.tolist()
        
        # Debug: Print first joint conversion for verification
        if joint_name == BEAT_GESTURE_JOINT_ORDER[0]:  # First joint (Spine)
            euler_deg = euler_degrees_full[start_idx:start_idx+3]
            euler_rad = euler_radians[start_idx:start_idx+3]
            axis_unnorm = axis_angle[start_idx:start_idx+3]
            mean_vals = mean_axis[start_idx:start_idx+3]
            std_vals = std_axis[start_idx:start_idx+3]
            
            print(f"\n      [DEBUG] First joint conversion ({joint_name}):")
            print(f"              1. Input (Euler deg):      [{euler_deg[0]:.6f}, {euler_deg[1]:.6f}, {euler_deg[2]:.6f}]")
            print(f"              2. Euler (radians):        [{euler_rad[0]:.6f}, {euler_rad[1]:.6f}, {euler_rad[2]:.6f}]")
            print(f"              3. Axis-angle (denorm):    [{axis_unnorm[0]:.6f}, {axis_unnorm[1]:.6f}, {axis_unnorm[2]:.6f}]")
            print(f"              4. Axis-angle mean:        [{mean_vals[0]:.6f}, {mean_vals[1]:.6f}, {mean_vals[2]:.6f}]")
            print(f"              5. Axis-angle std:         [{std_vals[0]:.6f}, {std_vals[1]:.6f}, {std_vals[2]:.6f}]")
            print(f"              6. Output (norm axis-ang): [{norm_vals[0]:.6f}, {norm_vals[1]:.6f}, {norm_vals[2]:.6f}]")
    
    return normalized


def save_custom_neutral_positions(normalized_angles: dict, output_path: str):
    """
    Save normalized axis-angle values as custom_neutral_positions YAML.
    
    Args:
        normalized_angles: Dict of joint_name -> [x_norm, y_norm, z_norm] in normalized axis-angle space
        output_path: Output YAML file path
    """
    # Save to file
    with open(output_path, 'w') as f:
        # Write header comments
        f.write('# Custom Neutral Positions for DiffSHEG Wrapper\n')
        f.write('# Generated from frozen skeleton pose\n')
        f.write('# These values are NORMALIZED AXIS-ANGLE and ready for use in embodiment_manager_config.yaml\n')
        f.write('# The model uses axis_angle=True, so custom_neutral_positions must be in this format.\n')
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
        
        # Write custom_neutral_positions section
        f.write('  custom_neutral_positions:\n')
        
        # Add joints in BEAT order with inline list format (flow style)
        for joint_name in BEAT_GESTURE_JOINT_ORDER:
            if joint_name in normalized_angles:
                # Format with 6 decimal places in flow style [x, y, z]
                angles = normalized_angles[joint_name]
                f.write(f'    {joint_name}: [{angles[0]:.6f}, {angles[1]:.6f}, {angles[2]:.6f}]\n')
    
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
    
    # HARDCODED: Must match co_speech_gesture_server.py exactly
    # The server uses opt.beat_cache_name which defaults to 'beat_4english_15_141'
    # from DiffSHEG/options/base_options.py
    # DO NOT change this unless you change the server configuration!
    HARDCODED_CACHE_NAME = 'beat_4english_15_141'
    
    # Remove --cache_name argument - no longer configurable to prevent mismatches
    # parser.add_argument(
    #     '--cache_name',
    #     type=str,
    #     default='beat_4english_15_141',
    #     help='Cache name for loading normalization statistics (default: beat_4english_15_141)'
    # )
    
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
    print(f"Cache name:  {HARDCODED_CACHE_NAME} (hardcoded to match server)")
    print(f"Output file: {args.output}")
    print("="*80)
    
    try:
        # Step 1: Load frozen pose
        print("\n[1/4] Loading frozen skeleton pose...")
        denormalized_angles = load_frozen_pose(args.input)
        print(f"      ✓ Loaded {len(denormalized_angles)} joints")
        
        # Step 2: Load normalization statistics
        print("\n[2/4] Loading normalization statistics...")
        print("      Model uses axis_angle=True, so loading axis-angle statistics")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mean_axis, std_axis = load_normalization_stats(HARDCODED_CACHE_NAME, script_dir)
        print(f"      ✓ Loaded axis-angle mean and std from cache '{HARDCODED_CACHE_NAME}'")
        print(f"      - Mean shape: {mean_axis.shape}")
        print(f"      - Std shape:  {std_axis.shape}")
        print(f"      - Sample mean values (first 3): {mean_axis[:3]}")
        print(f"      - Sample std values (first 3):  {std_axis[:3]}")
        
        # Step 3: Convert to normalized axis-angle
        print("\n[3/4] Converting Euler degrees → normalized axis-angle...")
        print("      This involves: Euler(deg) → Euler(rad) → axis-angle → normalize")
        normalized_angles = convert_euler_to_normalized_axis_angle(denormalized_angles, mean_axis, std_axis)
        print(f"      ✓ Converted {len(normalized_angles)} joints")
        
        # Step 4: Save output
        print("\n[4/4] Saving custom neutral positions...")
        save_custom_neutral_positions(normalized_angles, args.output)
        
        # Print summary
        print_summary(denormalized_angles, normalized_angles)
        
        # Print usage instructions
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n✓ SUCCESS: Converted frozen pose to NORMALIZED AXIS-ANGLE format")
        print(f"\n  Output file: {args.output}")
        print(f"  Format: Normalized axis-angle (matching model output space)")
        print(f"  Joints converted: {len(normalized_angles)}")
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
        print("\nIMPORTANT: The values in the config are NORMALIZED AXIS-ANGLE, not Euler degrees!")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
