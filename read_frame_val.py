import numpy as np
import sys
import os
import yaml

'''
Reference command:
python read_frame_val.py     --bvh_file /home/eeyifanshen/e2e_audio_LLM/SocialTaskImplementation/embodiment_manager/DiffSHEG/results/BestFGD_e999_ddim25_lastStepInterp/pid_6/gesture/bvh/2_scott_0_3_3.bvh     --cache_name beat_4english_15_141     --normalized     --output_yaml neutral_positions.yaml
'''



# BEAT skeleton joint order - must match diffsheg_realtime_wrapper.py and test_realtime_wrapper.py
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


def extract_denormalized_from_bvh(bvh_file_path, joint_names, frame_idx=0):
    """
    Extract denormalized joint angles (in degrees) from specified frame of BVH file.
    
    The BVH file contains DENORMALIZED EULER ANGLES in degrees.
    These are the final output from test_realtime_wrapper.py.
    
    Args:
        bvh_file_path: Path to the BVH file
        joint_names: List of joint names to extract (e.g., ['RightArm', 'RightForeArm'])
        frame_idx: Frame index to extract from (default: 0 = first frame)
    
    Returns:
        Dictionary mapping joint_name -> [x_rot, y_rot, z_rot] in degrees
    """
    
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse HIERARCHY to build joint order and channel count
    joint_channels = {}  # {joint_name: (start_channel_idx, num_channels)}
    channel_count = 0
    current_joint = None
    
    in_hierarchy = False
    for line in lines:
        if line.strip().startswith('HIERARCHY'):
            in_hierarchy = True
            continue
        
        if line.strip().startswith('MOTION'):
            in_hierarchy = False
            break
        
        if not in_hierarchy:
            continue
        
        line = line.strip()
        
        if line.startswith('ROOT '):
            current_joint = line.split()[1]
            joint_channels[current_joint] = (channel_count, 0)
        
        elif line.startswith('JOINT '):
            current_joint = line.split()[1]
            joint_channels[current_joint] = (channel_count, 0)
        
        elif line.startswith('CHANNELS'):
            parts = line.split()
            num_channels = int(parts[1])
            if current_joint:
                start_idx = channel_count
                joint_channels[current_joint] = (start_idx, num_channels)
                channel_count += num_channels
    
    # Extract frame data
    frame_start_idx = None
    num_frames = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('Frames:'):
            frame_start_idx = i + 2
            # Extract number of frames from the line before
            frames_line = lines[i].strip()
            num_frames = int(frames_line.split(':')[1].strip())
            break
    
    if frame_start_idx is None:
        raise ValueError("Could not find MOTION data in BVH file")
    
    # Validate frame index
    if frame_idx < 0 or frame_idx >= num_frames:
        raise ValueError(f"Frame index {frame_idx} out of range [0, {num_frames-1}]")
    
    # Get the requested frame line
    frame_line = lines[frame_start_idx + frame_idx].strip()
    frame_values = [float(x) for x in frame_line.split()]
    
    # Extract positions for requested joints
    positions = {}
    
    for joint_name in joint_names:
        if joint_name not in joint_channels:
            print(f"⚠️  Joint '{joint_name}' not found in BVH hierarchy")
            continue
        
        start_idx, num_channels = joint_channels[joint_name]
        
        if num_channels == 3:  # Standard Xrotation Yrotation Zrotation
            x_rot = frame_values[start_idx]
            y_rot = frame_values[start_idx + 1]
            z_rot = frame_values[start_idx + 2]
            
            positions[joint_name] = [x_rot, y_rot, z_rot]
        else:
            print(f"⚠️  Joint '{joint_name}' has {num_channels} channels (expected 3)")
    
    return positions


def normalize_euler_angles(denormalized_angles, mean_pose, std_pose):
    """
    Convert denormalized Euler angles back to normalized space.
    
    This is the INVERSE of the denormalization process used in test_realtime_wrapper.py:
    - test_realtime_wrapper: denorm = norm * std_pose + mean_pose
    - This function: norm = (denorm - mean_pose) / std_pose
    
    Args:
        denormalized_angles: Dict of joint_name -> [x, y, z] in degrees (from BVH)
        mean_pose: numpy array of shape (141,) - Euler mean from bvh_mean.npy
        std_pose: numpy array of shape (141,) - Euler std from bvh_std.npy
    
    Returns:
        Dict of joint_name -> [x_norm, y_norm, z_norm] in normalized space
    """
    normalized = {}
    
    for joint_name, angles in denormalized_angles.items():
        # Find this joint's index in BEAT_GESTURE_JOINT_ORDER
        if joint_name not in BEAT_GESTURE_JOINT_ORDER:
            print(f"⚠️  Joint '{joint_name}' not in BEAT_GESTURE_JOINT_ORDER")
            continue
        
        joint_idx = BEAT_GESTURE_JOINT_ORDER.index(joint_name)
        start_idx = joint_idx * 3
        
        # Extract denormalized values
        denorm = np.array(angles, dtype=np.float32)
        
        # Get corresponding mean/std for this joint (3 channels)
        mean_vals = mean_pose[start_idx:start_idx+3]
        std_vals = std_pose[start_idx:start_idx+3]
        
        # Normalize: (denorm - mean) / std
        norm_vals = (denorm - mean_vals) / std_vals
        
        normalized[joint_name] = norm_vals.tolist()
    
    return normalized


# Usage - Extract neutral positions from BVH and format for config
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract neutral positions from BVH file for DiffSHEG wrapper config')
    parser.add_argument('--bvh_file', type=str, required=True, help='Path to BVH file')
    parser.add_argument('--frame', type=int, default=0, help='Frame index to extract from (default: 0 = first frame)')
    parser.add_argument('--cache_name', type=str, default='beat_4english_15_141', 
                        help='Cache name for loading normalization statistics (default: beat_4english_15_141)')
    parser.add_argument('--output_yaml', type=str, default=None, help='Output YAML file (optional)')
    parser.add_argument('--normalized', action='store_true', default=False,
                        help='Convert extracted BVH values to normalized space (required for wrapper config)')
    
    args = parser.parse_args()
    
    # All joints in BEAT that can be masked
    all_beat_joints = BEAT_GESTURE_JOINT_ORDER.copy()
    
    print("=" * 80)
    print(f"EXTRACTING NEUTRAL POSITIONS FROM BVH: {args.bvh_file}")
    print(f"Frame index: {args.frame}")
    print(f"Cache name: {args.cache_name}")
    print(f"Normalized output: {args.normalized}")
    print("=" * 80)
    
    try:
        # Extract denormalized values from BVH
        denorm_pos = extract_denormalized_from_bvh(args.bvh_file, all_beat_joints, frame_idx=args.frame)
        
        # Convert to normalized space if requested
        if args.normalized:
            print("\nLoading normalization statistics...")
            
            # Construct paths - matching co_speech_gesture_server.py
            script_dir = os.path.dirname(os.path.abspath(__file__))
            stats_dir = os.path.join(script_dir, "data", "BEAT", "beat_cache", args.cache_name, "train")
            
            mean_euler_path = os.path.join(stats_dir, "bvh_rot", "bvh_mean.npy")
            std_euler_path = os.path.join(stats_dir, "bvh_rot", "bvh_std.npy")
            
            if not os.path.exists(mean_euler_path):
                raise FileNotFoundError(f"Mean statistics not found at: {mean_euler_path}")
            if not os.path.exists(std_euler_path):
                raise FileNotFoundError(f"Std statistics not found at: {std_euler_path}")
            
            # Load statistics
            mean_pose = np.load(mean_euler_path)
            std_pose = np.load(std_euler_path)
            
            print(f"✓ Loaded mean_pose: shape {mean_pose.shape}")
            print(f"✓ Loaded std_pose: shape {std_pose.shape}")
            
            # Convert to normalized space
            print("\nConverting from denormalized to normalized space...")
            print("  Formula: normalized = (denormalized - mean) / std")
            
            neutral_pos = normalize_euler_angles(denorm_pos, mean_pose, std_pose)
            
            print("✓ Conversion complete - values are now in NORMALIZED space")
            space_label = "NORMALIZED"
        else:
            neutral_pos = denorm_pos
            space_label = "DENORMALIZED"
        
        print("\n" + "=" * 80)
        print(f"NEUTRAL POSITIONS FOR DIFFSHEG WRAPPER CONFIG ({space_label} space)")
        print("=" * 80)
        
        # Format for YAML config
        yaml_output = "custom_neutral_positions:\n"
        for joint_name, angles in neutral_pos.items():
            if isinstance(angles[0], float):
                yaml_output += f"  {joint_name}: [{angles[0]:.6f}, {angles[1]:.6f}, {angles[2]:.6f}]\n"
            else:
                yaml_output += f"  {joint_name}: {angles}\n"
        
        print("\nYAML format (copy to embodiment_manager_config.yaml):")
        print("-" * 80)
        print(yaml_output)
        
        # Save to file if requested
        if args.output_yaml:
            with open(args.output_yaml, 'w') as f:
                f.write(yaml_output)
            print(f"\n✓ Saved to {args.output_yaml}")
        
        print("=" * 80)
        print("IMPORTANT: SPACE MISMATCH WARNING")
        print("-" * 80)
        if not args.normalized:
            print("""
⚠️  BVH files contain DENORMALIZED Euler angles in degrees.
⚠️  The wrapper applies neutral positions in NORMALIZED space.
⚠️  Use --normalized flag to convert to normalized space!

Example:
    python read_frame_val.py --bvh_file gesture.bvh --normalized

Without --normalized, the wrapper will apply denormalized values as if they 
were normalized, causing the skeleton to twist unnaturally!
""")
        else:
            print("""
✓ Normalized space values can be applied directly in wrapper config.
✓ The wrapper will use these as neutral positions in its normalized space.
✓ Make sure the cache_name matches your DiffSHEG checkpoint!
""")
        
        print("USAGE IN CONFIG:")
        print("-" * 80)
        print("""
Add the following to embodiment_manager_config.yaml under co_speech_gestures section:

co_speech_gestures:
  # ... other config ...
  custom_neutral_positions:
    RightShoulder: [x.xxxxxx, y.yyyyyy, z.zzzzzz]
    RightArm: [x.xxxxxx, y.yyyyyy, z.zzzzzz]
    # ... etc (all 47 joints)
""")
        print("=" * 80)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()