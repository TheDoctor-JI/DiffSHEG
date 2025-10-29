"""
Test script for DiffSHEG Realtime Wrapper
"""
import os
import sys
import time
import numpy as np
import torch
import librosa
from pathlib import Path
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from options.train_options import TrainCompOptions
from models import MotionTransformer, UniDiffuser
from trainers import DDPMTrainer_beat
from diffsheg_realtime_wrapper import DiffSHEGRealtimeWrapper
import datasets.rotation_converter as rot_cvt


class WaypointCollector:
    """
    Collects waypoints during realtime generation and exports them to BVH+JSON format.
    
    Handles timing alignment: waypoints have timestamps relative to audio start,
    but gesture generation may not cover the entire audio duration (due to start_margin).
    """
    
    def __init__(self, split_pos=141, gesture_fps=15):
        self.waypoints = []
        self.split_pos = split_pos  # Split position between gesture and expression
        self.gesture_fps = gesture_fps
        self.first_timestamp = None  # Track when gestures actually start
        self.last_timestamp = None   # Track when gestures actually end
    
    def collect(self, waypoint):
        """Store each waypoint as it arrives"""
        self.waypoints.append({
            'index': waypoint.waypoint_index,
            'timestamp': waypoint.timestamp,  # Time in seconds from utterance start
            'gesture_data': waypoint.gesture_data.copy()  # shape (192,)
        })
        
        # Track timing bounds
        if self.first_timestamp is None:
            self.first_timestamp = waypoint.timestamp
        self.last_timestamp = waypoint.timestamp
        
        print(f"[WAYPOINT] Collected waypoint {waypoint.waypoint_index} at t={waypoint.timestamp:.3f}s - shape: {waypoint.gesture_data.shape}")
    
    def save_to_files(self, output_dir, trainer, audio_duration):
        """
        Convert collected waypoints to BVH + JSON format for Blender visualization.
        
        Handles timing alignment:
        - Waypoints have timestamps relative to audio start
        - May not cover entire audio duration (start_margin delay)
        - Fills missing frames with neutral/zero poses
        
        Args:
            output_dir: Directory to save output files
            trainer: DDPMTrainer_beat instance with conversion methods and normalization stats
            audio_duration: Total duration of audio in seconds
        """
        if len(self.waypoints) == 0:
            print("[EXPORT] No waypoints collected!")
            return
        
        print(f"\n[EXPORT] Processing {len(self.waypoints)} waypoints...")
        print(f"[EXPORT] Waypoint timing: first={self.first_timestamp:.3f}s, last={self.last_timestamp:.3f}s")
        print(f"[EXPORT] Audio duration: {audio_duration:.3f}s")
        
        # Sort waypoints by index to ensure correct order
        self.waypoints.sort(key=lambda x: x['index'])
        
        # Calculate total frames needed to cover the entire audio
        total_audio_frames = int(np.ceil(audio_duration * self.gesture_fps))
        
        # Create frame array for entire audio duration, initialized with zeros
        # This will be filled with actual waypoint data where available
        all_poses = np.zeros((total_audio_frames, 192), dtype=np.float32)
        
        # Fill in the waypoint data at appropriate frame indices
        for waypoint in self.waypoints:
            # Convert timestamp to frame index
            frame_idx = int(np.round(waypoint['timestamp'] * self.gesture_fps))
            
            # Ensure we don't exceed array bounds
            if 0 <= frame_idx < total_audio_frames:
                all_poses[frame_idx] = waypoint['gesture_data']
        
        # For frames without waypoints (before first waypoint or after last),
        # we can either leave them as zero or copy the nearest valid frame
        # Let's use forward-fill and backward-fill strategy
        first_frame_idx = int(np.round(self.first_timestamp * self.gesture_fps))
        last_frame_idx = int(np.round(self.last_timestamp * self.gesture_fps))
        
        # Forward fill: copy first valid frame to all frames before it
        if first_frame_idx > 0:
            all_poses[:first_frame_idx] = all_poses[first_frame_idx]
            print(f"[EXPORT] Filled frames 0-{first_frame_idx-1} with first waypoint pose")
        
        # Backward fill: copy last valid frame to all frames after it
        if last_frame_idx < total_audio_frames - 1:
            all_poses[last_frame_idx+1:] = all_poses[last_frame_idx]
            print(f"[EXPORT] Filled frames {last_frame_idx+1}-{total_audio_frames-1} with last waypoint pose")
        
        # Split into gesture (141) and expression (51)
        gestures = all_poses[:, :self.split_pos]      # (total_audio_frames, 141)
        expressions = all_poses[:, self.split_pos:]   # (total_audio_frames, 51)
        
        print(f"[EXPORT] Split poses: gestures={gestures.shape}, expressions={expressions.shape}")
        
        # Load normalization statistics for gestures
        # These are the same stats used during training
        mean_pose_path = f"data/BEAT/beat_cache/{trainer.opt.beat_cache_name}/train/bvh_rot/bvh_mean.npy"
        std_pose_path = f"data/BEAT/beat_cache/{trainer.opt.beat_cache_name}/train/bvh_rot/bvh_std.npy"
        
        mean_pose = torch.from_numpy(np.load(mean_pose_path)).float()
        std_pose = torch.from_numpy(np.load(std_pose_path)).float()
        
        # Convert gestures to BVH format (body motion)
        # Gestures are in normalized space and need denormalization
        gestures_tensor = torch.from_numpy(gestures).unsqueeze(0)  # (1, T, 141)
        
        # Denormalize using dataset statistics
        denorm_gestures = gestures_tensor * std_pose + mean_pose
        denorm_gestures_np = denorm_gestures.squeeze(0).numpy()  # (T, 141)
        
        # Convert to degrees (gestures are in Euler angles)
        denorm_gestures_np = denorm_gestures_np * (180 / np.pi)
        
        print(f"[EXPORT] Denormalized gestures: shape={denorm_gestures_np.shape}")
        
        # Save BVH file
        bvh_dir = os.path.join(output_dir, 'bvh')
        os.makedirs(bvh_dir, exist_ok=True)
        trainer.result2target_vis(
            denorm_gestures_np, 
            bvh_dir, 
            'realtime_output.bvh'
        )
        print(f"[EXPORT] Saved BVH to {bvh_dir}/realtime_output.bvh")
        
        # Convert expressions to JSON format (facial blendshapes)
        expressions_expanded = np.expand_dims(expressions, 0)  # (1, T, 51)
        json_dir = os.path.join(output_dir, 'face_json')
        os.makedirs(json_dir, exist_ok=True)
        trainer.write_face_json(
            expressions_expanded, 
            os.path.join(json_dir, 'realtime_output.json')
        )
        print(f"[EXPORT] Saved JSON to {json_dir}/realtime_output.json")
        
        print(f"\n[EXPORT] Export complete!")
        print(f"[EXPORT] Total frames: {total_audio_frames} ({audio_duration:.2f}s @ {self.gesture_fps} FPS)")
        print(f"[EXPORT] Waypoint coverage: frames {first_frame_idx}-{last_frame_idx}")


# Global waypoint collector
waypoint_collector = None


def waypoint_handler(waypoint):
    """
    Callback function to handle waypoint execution.
    
    Args:
        waypoint: GestureWaypoint object containing waypoint information
    """
    if waypoint_collector is not None:
        waypoint_collector.collect(waypoint)


def build_model(opt):
    """Build DiffSHEG model"""
    if opt.unidiffuser:
        encoder = UniDiffuser(
            opt=opt,
            input_feats=opt.net_dim_pose,
            audio_dim=opt.audio_dim,
            aud_latent_dim=opt.audio_latent_dim,
            style_dim=opt.style_dim,
            num_frames=opt.n_poses,
            num_layers=opt.num_layers,
            latent_dim=opt.latent_dim,
            no_clip=opt.no_clip,
            no_eff=opt.no_eff,
            pe_type=opt.PE)
    else:
        encoder = MotionTransformer(
            opt=opt,
            input_feats=opt.net_dim_pose,
            audio_dim=opt.audio_dim,
            style_dim=opt.style_dim,
            num_frames=opt.n_poses,
            num_layers=opt.num_layers,
            latent_dim=opt.latent_dim,
            no_clip=opt.no_clip,
            no_eff=opt.no_eff,
            pe_type=opt.PE)
    return encoder


def setup_beat_config(opt):
    """Configure options for BEAT dataset"""
    opt.data_root = 'data/BEAT'
    opt.fps = 15
    opt.net_dim_pose = 192
    opt.split_pos = 141
    opt.dim_pose = 141
    if opt.remove_hand:
        opt.dim_pose = 33
    opt.expression_dim = 51
    
    if opt.expression_only or opt.gesCondition_expression_only:
        opt.net_dim_pose = opt.expression_dim
        opt.e_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/weights/face_300.bin'
    elif opt.gesture_only or opt.expCondition_gesture_only != None or opt.textExpEmoCondition_gesture_only:
        opt.net_dim_pose = opt.dim_pose
        if opt.axis_angle:
            opt.e_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/weights/ges_axis_angle_300.bin'
        else:
            opt.e_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/weights/ae_300.bin'
    else:
        opt.net_dim_pose = opt.dim_pose + opt.expression_dim
        if opt.axis_angle:
            opt.e_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/weights/GesAxisAngle_Face_300.bin'
        else:
            raise NotImplementedError
    
    opt.audio_dim = 128
    if opt.use_aud_feat:
        opt.audio_dim = 1024
    opt.style_dim = 30
    opt.speaker_dim = 30
    opt.word_index_num = 5793
    opt.word_dims = 300
    opt.word_f = 128
    opt.emotion_f = 8
    opt.emotion_dims = 8
    opt.freeze_wordembed = False
    opt.hidden_size = 256
    opt.n_layer = 4
    
    if opt.n_poses == 150:
        opt.stride = 50
    elif opt.n_poses == 34:
        opt.stride = 10
    opt.pose_fps = 15
    opt.vae_length = 300
    opt.new_cache = False
    opt.audio_norm = False
    opt.facial_norm = True
    opt.pose_norm = True
    
    return opt


def load_audio_file(audio_path, target_sr=16000):
    """Load audio file and return as chunks"""
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    return audio, sr


def audio_to_chunks(audio, sr, chunk_duration=0.04):
    """
    Convert audio array to chunks matching the system's format.
    
    Args:
        audio: Float audio array (-1.0 to 1.0)
        sr: Sample rate
        chunk_duration: Chunk duration in seconds (0.04s = 40ms for system audio)
    
    Returns:
        List of chunks as list of byte values (0-255), matching app.py pipeline
    """
    chunk_samples = int(sr * chunk_duration)
    audio_int16 = (audio * 32767).astype(np.int16)
    
    chunks = []
    for i in range(0, len(audio_int16), chunk_samples):
        chunk = audio_int16[i:i + chunk_samples]
        # Convert to bytes (s16le encoding)
        chunk_bytes = chunk.tobytes()
        # Convert bytes to list of byte values (0-255) - matching app.py format
        # where audio arrives as list of integers and gets converted back via bytes()
        chunk_list = list(chunk_bytes)
        chunks.append(chunk_list)
    
    return chunks


def main():
    global waypoint_collector
    
    # Parse arguments (mimicking the bash script)
    parser = TrainCompOptions()
    
    # Set up arguments to match inference_custom_audio_beat.sh
    sys.argv = [
        'test_realtime_wrapper.py',
        '--dataset_name', 'beat',
        '--name', 'beat_GesExpr_unify_addHubert_encodeHubert_mlpIncludeX_condRes_LN',
        '--n_poses', '34',
        '--ddim',
        '--ckpt', 'fgd_best.tar',
        '--timestep_respacing', 'ddim25',
        '--overlap_len', '4',
        '--mode', 'test_custom_audio',
        '--jump_n_sample', '2',
    ]
    
    opt = parser.parse()
    opt = setup_beat_config(opt)
    
    # Set device
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt.gpu_id = 0
    print(f"Using device: {opt.device}")
    
    # Build model
    print("Building model...")
    model = build_model(opt)
    model = model.to(opt.device)
    
    # Initialize trainer (which handles model loading and normalization stats)
    print("Initializing trainer...")
    trainer = DDPMTrainer_beat(opt, model, eval_model=None)
    
    # Initialize waypoint collector
    print("Initializing waypoint collector...")
    waypoint_collector = WaypointCollector(
        split_pos=opt.split_pos,  # 141 for gesture/expression split
        gesture_fps=15  # BEAT dataset uses 15 FPS
    )
    
    # Initialize wrapper
    print("Initializing realtime wrapper...")
    wrapper = DiffSHEGRealtimeWrapper(
        diffsheg_model=trainer,
        opt=opt,
        default_start_margin=2.5,  # 0.5s margin
        audio_sr=16000,  # From config
        device=opt.device,
        cleanup_timeout=2.0,
        waypoint_callback=waypoint_handler
    )
    
    # Start wrapper threads
    print("Starting wrapper threads...")
    wrapper.start()
    
    # Load test audio
    audio_path = 'audios/2_scott_0_3_3.wav'
    print(f"Loading test audio from {audio_path}...")
    audio, sr = load_audio_file(audio_path, target_sr=16000)
    
    audio_duration = len(audio) / sr
    print(f"Audio duration: {audio_duration:.2f}s")
    
    # Convert to chunks
    chunk_duration = 0.04  # 40ms chunks (system audio chunk size)
    print(f"Converting to {chunk_duration}s chunks...")
    chunks = audio_to_chunks(audio, sr, chunk_duration)
    print(f"Total chunks: {len(chunks)}")
    
    # Stream chunks faster than realtime
    utterance_id = 1
    playback_speed = 5.0  # faster than realtime
    chunk_interval = chunk_duration / playback_speed
    
    print(f"\nStreaming chunks at {playback_speed}x speed (chunk every {chunk_interval:.3f}s)...")
    print(f"Estimated streaming time: {len(chunks) * chunk_interval:.2f}s")
    
    start_time = time.time()
    for i, chunk in enumerate(chunks):
        wrapper.add_audio_chunk(
            utterance_id=utterance_id,
            chunk_index=i,
            audio_data=chunk,
            duration=chunk_duration
        )
        
        if (i + 1) % 10 == 0:
            print(f"Sent chunk {i + 1}/{len(chunks)}")
        
        if i < len(chunks) - 1:  # Don't sleep after last chunk
            time.sleep(chunk_interval)
    
    elapsed = time.time() - start_time
    print(f"\nAll chunks sent in {elapsed:.2f}s")
    
    # Wait for processing to complete
    print("\nWaiting for gesture generation and playback to complete...")
    # Wait for audio duration + cleanup timeout
    wait_time = audio_duration + wrapper.cleanup_timeout + 1.0
    print(f"Waiting {wait_time:.1f}s...")
    time.sleep(wait_time)
    
    # Stop wrapper
    print("\nStopping wrapper...")
    wrapper.stop()
    
    # Export waypoints to visualization format
    print("\n" + "="*60)
    print("Exporting waypoints to visualization format...")
    print("="*60)
    
    output_dir = 'results/realtime_test'
    os.makedirs(output_dir, exist_ok=True)
    
    waypoint_collector.save_to_files(
        output_dir=output_dir,
        trainer=trainer,
        audio_duration=audio_duration
    )
    
    # Copy audio file to output directory for convenience
    audio_output_path = os.path.join(output_dir, os.path.basename(audio_path))
    shutil.copy(audio_path, audio_output_path)
    print(f"[EXPORT] Copied audio to {audio_output_path}")
    
    print("\n" + "="*60)
    print("Next Steps for Blender Visualization:")
    print("="*60)
    print("1. Open assets/beat_visualize.blend in Blender")
    print("2. Set paths in the Blender script:")
    print(f"   - BVH: {os.path.abspath(os.path.join(output_dir, 'bvh', 'realtime_output.bvh'))}")
    print(f"   - JSON: {os.path.abspath(os.path.join(output_dir, 'face_json', 'realtime_output.json'))}")
    print(f"   - Audio: {os.path.abspath(audio_output_path)}")
    print("3. Run the script in Blender to render video")
    print("="*60)
    
    print("\nTest complete!")


if __name__ == "__main__":
    main()
