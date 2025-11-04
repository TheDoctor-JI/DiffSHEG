import numpy as np

def extract_neutral_positions_from_bvh(bvh_file_path, joint_names):
    """
    Extract neutral positions (first frame rotations) from BVH file for specified joints.
    
    Args:
        bvh_file_path: Path to the BVH file
        joint_names: List of joint names to extract (e.g., ['RightArm', 'RightForeArm'])
    
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
            joint_channels[current_joint] = (channel_count, 0)  # Will update after CHANNELS line
        
        elif line.startswith('JOINT '):
            current_joint = line.split()[1]
            joint_channels[current_joint] = (channel_count, 0)  # Will update after CHANNELS line
        
        elif line.startswith('CHANNELS'):
            parts = line.split()
            num_channels = int(parts[1])
            if current_joint:
                start_idx = channel_count
                joint_channels[current_joint] = (start_idx, num_channels)
                channel_count += num_channels
    
    # Extract first frame data
    frame_start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('Frames:'):
            # First frame is two lines after "Frames:" (after "Frame Time:" line)
            frame_start_idx = i + 2
            break
    
    if frame_start_idx is None:
        raise ValueError("Could not find MOTION data in BVH file")
    
    first_frame_line = lines[frame_start_idx].strip()
    frame_values = [float(x) for x in first_frame_line.split()]
    
    # Extract neutral positions for requested joints
    neutral_positions = {}
    
    for joint_name in joint_names:
        if joint_name not in joint_channels:
            print(f"⚠️  Joint '{joint_name}' not found in BVH hierarchy")
            continue
        
        start_idx, num_channels = joint_channels[joint_name]
        
        if num_channels == 3:  # Standard Xrotation Yrotation Zrotation
            x_rot = frame_values[start_idx]
            y_rot = frame_values[start_idx + 1]
            z_rot = frame_values[start_idx + 2]
            
            neutral_positions[joint_name] = [x_rot, y_rot, z_rot]
            
            print(f"✓ {joint_name}: X={x_rot:.2f}°, Y={y_rot:.2f}°, Z={z_rot:.2f}°")
        else:
            print(f"⚠️  Joint '{joint_name}' has {num_channels} channels (expected 3)")
    
    return neutral_positions


# Usage
bvh_file = '/home/eeyifanshen/e2e_audio_LLM/SocialTaskImplementation/embodiment_manager/DiffSHEG/results/BestFGD_e999_ddim25_lastStepInterp/pid_6/gesture/bvh/2_scott_0_3_3.bvh'
masked_joints = ['RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 
                 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand']

neutral_pos = extract_neutral_positions_from_bvh(bvh_file, masked_joints)

# Print in YAML format for your config
print("\ncustom_neutral_position_for_masked_joint:")
for joint_name, angles in neutral_pos.items():
    print(f"  {joint_name}: [{angles[0]:.2f}, {angles[1]:.2f}, {angles[2]:.2f}]")