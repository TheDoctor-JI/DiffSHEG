"""
Test script for DiffSHEG Realtime Wrapper

TESTING MODES:
==============

1. STREAMING MODE (default):
   - DiffSHEGRealtimeWrapper.NON_STREAMING_SANITY_CHECK = False
   - Generates gestures in real-time as audio chunks arrive
   - Uses incremental windowing with audio truncation for streaming
   
2. NON-STREAMING SANITY CHECK MODE:
   - DiffSHEGRealtimeWrapper.NON_STREAMING_SANITY_CHECK = True
   - Waits for complete audio before generating
   - Uses same official generation pipeline but processes window-by-window
   - Useful for debugging without streaming complexity

Both modes use the EXACT same official DiffSHEG generation pipeline:
- Official mel spectrogram extraction (librosa.feature.melspectrogram)
- Official HuBERT feature extraction (get_hubert_from_16k_speech_long)
- Official trainer.generate_batch() method with proper inpainting
- Audio truncation strategy: pretend no audio exists beyond current window

Export uses official trainer methods for conversion to BVH/JSON format.
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
import torch.nn.functional as F
from trainers.ddpm_beat_trainer import get_hubert_from_16k_speech_long
from transformers import Wav2Vec2Processor, HubertModel


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
        
        # For frames without waypoints (before first waypoint or after last):
        # - Before first waypoint: Keep as zeros (neutral pose in normalized space)
        # - After last waypoint: Copy last valid frame to hold final pose
        # 
        # CRITICAL: Zeros in normalized space = neutral pose!
        # Since waypoints are in normalized space (model output after mean subtraction),
        # zeros represent the mean pose (neutral standing position).
        first_frame_idx = int(np.round(self.first_timestamp * self.gesture_fps))
        last_frame_idx = int(np.round(self.last_timestamp * self.gesture_fps))
        
        # Keep frames before first waypoint as zeros (neutral pose)
        if first_frame_idx > 0:
            # all_poses[:first_frame_idx] already zeros from initialization
            print(f"[EXPORT] Frames 0-{first_frame_idx-1} kept as zeros (neutral pose in normalized space)")
        
        # Backward fill: copy last valid frame to all frames after it
        if last_frame_idx < total_audio_frames - 1:
            all_poses[last_frame_idx+1:] = all_poses[last_frame_idx]
            print(f"[EXPORT] Filled frames {last_frame_idx+1}-{total_audio_frames-1} with last waypoint pose")
        
        # === CRITICAL: Split FIRST, then handle axis-angle to Euler conversion ===
        # Model outputs normalized axis-angle representation (opt.axis_angle=True by default)
        # We need to: denormalize axis-angle -> convert to Euler -> denormalize Euler -> BVH
        
        # Split into gesture (141) and expression (51) while still normalized
        gestures_normalized = all_poses[:, :self.split_pos]      # (T, 141) - still normalized axis-angle
        expressions_normalized = all_poses[:, self.split_pos:]   # (T, 51) - still normalized
        
        print(f"[EXPORT] Split normalized poses: gestures={gestures_normalized.shape}, expressions={expressions_normalized.shape}")
        
        # Check if model uses axis-angle representation (default is True)
        use_axis_angle = getattr(trainer.opt, 'axis_angle', True)
        
        if use_axis_angle:
            print(f"[EXPORT] Converting axis-angle -> Euler angles (opt.axis_angle=True)")
            
            # Load axis-angle normalization statistics (motion_mean/motion_std from dataset)
            # These are stored in train/axis_angle_mean.npy and train/axis_angle_std.npy
            mean_axis_path = f"data/BEAT/beat_cache/{trainer.opt.beat_cache_name}/train/axis_angle_mean.npy"
            std_axis_path = f"data/BEAT/beat_cache/{trainer.opt.beat_cache_name}/train/axis_angle_std.npy"
            
            motion_mean = torch.from_numpy(np.load(mean_axis_path)).float()  # (141,) - axis-angle mean
            motion_std = torch.from_numpy(np.load(std_axis_path)).float()    # (141,) - axis-angle std
            
            # Load Euler angle statistics for final output (mean_pose/std_pose)
            mean_euler_path = f"data/BEAT/beat_cache/{trainer.opt.beat_cache_name}/train/bvh_rot/bvh_mean.npy"
            std_euler_path = f"data/BEAT/beat_cache/{trainer.opt.beat_cache_name}/train/bvh_rot/bvh_std.npy"
            
            mean_pose = torch.from_numpy(np.load(mean_euler_path)).float()  # (141,) - Euler mean
            std_pose = torch.from_numpy(np.load(std_euler_path)).float()    # (141,) - Euler std
            
            print(f"[EXPORT] Loaded stats: motion_mean/std={motion_mean.shape}, mean_pose/std_pose={mean_pose.shape}")
            
            # Step 1: Denormalize from axis-angle normalized space
            gestures_tensor = torch.from_numpy(gestures_normalized).unsqueeze(0)  # (1, T, 141)
            denorm_axis_angle = gestures_tensor * motion_std + motion_mean  # Denormalize to axis-angle
            
            # Step 2: Convert axis-angle to Euler angles (radians)
            B, T, C = denorm_axis_angle.shape
            euler_rad = rot_cvt.axis_angle_to_euler_angles(denorm_axis_angle.reshape(B, T, C//3, 3)).reshape(B, T, C)
            
            # Step 3: Convert radians to degrees
            euler_deg = euler_rad * (180 / np.pi)
            
            # Step 4: Re-normalize in Euler space and denormalize for final output
            euler_normalized = (euler_deg - mean_pose) / std_pose
            denorm_gestures = euler_normalized * std_pose + mean_pose  # Final Euler angles in degrees
            denorm_gestures_np = denorm_gestures.squeeze(0).numpy()  # (T, 141)
            
            print(f"[EXPORT] Converted axis-angle -> Euler angles (degrees): shape={denorm_gestures_np.shape}")
        else:
            print(f"[EXPORT] Using Euler angles directly (opt.axis_angle=False)")
            
            # Load Euler angle normalization statistics
            mean_euler_path = f"data/BEAT/beat_cache/{trainer.opt.beat_cache_name}/train/bvh_rot/bvh_mean.npy"
            std_euler_path = f"data/BEAT/beat_cache/{trainer.opt.beat_cache_name}/train/bvh_rot/bvh_std.npy"
            
            mean_pose = torch.from_numpy(np.load(mean_euler_path)).float()  # (141,)
            std_pose = torch.from_numpy(np.load(std_euler_path)).float()    # (141,)
            
            # Denormalize Euler angles directly
            gestures_tensor = torch.from_numpy(gestures_normalized).unsqueeze(0)  # (1, T, 141)
            denorm_gestures = gestures_tensor * std_pose + mean_pose  # Denormalize Euler angles
            denorm_gestures_np = denorm_gestures.squeeze(0).numpy()  # (T, 141)
        
        print(f"[EXPORT] Denormalized gestures: shape={denorm_gestures_np.shape}")
        
        # Save BVH file for gestures
        bvh_dir = os.path.join(output_dir, 'bvh')
        os.makedirs(bvh_dir, exist_ok=True)
        trainer.result2target_vis(
            denorm_gestures_np, 
            bvh_dir, 
            'realtime_output.bvh'
        )
        print(f"[EXPORT] Saved BVH to {bvh_dir}/realtime_output.bvh")
        
        # === Expression Processing ===
        # Convert expressions to JSON format (facial blendshapes)
        # Note: write_face_json expects NORMALIZED data and will denormalize it internally
        # expressions_normalized is already in normalized form (as split from model output)
        expressions_expanded = np.expand_dims(expressions_normalized, 0)  # (1, T, 51)
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
    opt.train_data_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/train/'
    opt.val_data_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/val/'
    opt.test_data_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/test/'
    opt.mean_pose_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/train/'
    opt.std_pose_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/train/'
    opt.multi_length_training = [1.0]
    opt.audio_rep = 'wave16k'
    opt.facial_rep = 'facial52'
    opt.speaker_id = 'id'
    opt.pose_rep = 'bvh_rot'
    opt.word_rep = 'text'
    
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


def official_export_logic(audio_path, trainer, opt, output_dir, pre_generated_motion=None):
    """
    Use the EXACT same export logic as runner.py test_custom_aud().
    
    Args:
        audio_path: Path to the audio file
        trainer: DDPMTrainer_beat instance
        opt: Options/config object
        output_dir: Output directory for results
        pre_generated_motion: Optional pre-generated motion tensor (B, T, 192) to skip generation.
                             If provided, only the export logic is used (conversion + saving).
                             If None, full generation pipeline is run.
    """
    print("\n[OFFICIAL EXPORT] Using exact same export logic as runner.py...")
    
    # Get audio name for later use
    name = os.path.basename(audio_path)
    
    if pre_generated_motion is None:
        # Full generation pipeline
        print("[OFFICIAL EXPORT] No pre-generated motion provided, running full generation...")
        
        # Load HuBERT models (required for addHubert=True)
        print("[OFFICIAL EXPORT] Loading HuBERT models...")
        wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        hubert_model = hubert_model.to(opt.device)
        
        # Person ID (use speaker 2 like in the official script)
        p_id_ori = 2
        p_id = p_id_ori - 1
        p_id_tensor = torch.ones((1, 1)) * p_id
        p_id_tensor = trainer.one_hot(p_id_tensor, opt.speaker_dim).detach().to(opt.device)
        
        # Load audio
        sr = 16000
        print(f"[OFFICIAL EXPORT] Loading audio: {name}")
        
        if name.endswith(".wav"):
            aud_ori, sr = librosa.load(audio_path)
        elif name.endswith(".npy"):
            aud_ori = np.load(audio_path)
        
        # Resample for mel spectrogram
        aud = librosa.resample(aud_ori, orig_sr=sr, target_sr=18000)
        
        # Extract mel spectrogram features (same as official runner)
        mel = librosa.feature.melspectrogram(y=aud, sr=18000, hop_length=1200, n_mels=128)
        mel = mel[..., :-1]
        audio_emb = torch.from_numpy(np.swapaxes(mel, -1, -2))
        audio_emb = audio_emb.unsqueeze(0).to(opt.device)
        
        B, N, _ = audio_emb.shape
        C = opt.net_dim_pose
        motions = torch.zeros((B, audio_emb.shape[-2], C)).to(opt.device)
        
        # Window generation (same as official runner)
        def get_windows(x, size, step):
            if isinstance(x, dict):
                out = {}
                for key in x.keys():
                    out[key] = get_windows(x[key], size, step)
                out_dict_list = []
                for i in range(len(out[list(out.keys())[0]])):
                    out_dict_list.append({key: out[key][i] for key in out.keys()})
                return out_dict_list
            else:
                seq_len = x.shape[1]
                if seq_len <= size:
                    return [x]
                else:
                    win_num = (seq_len - (size-step)) / float(step)
                    out = [x[:, mm*step : mm*step + size, ...] for mm in range(int(win_num))]
                    if win_num - int(win_num) != 0:
                        out.append(x[:, int(win_num)*step:, ...])  
                    return out
        
        window_step = opt.n_poses - opt.overlap_len
        audio_emb_list = get_windows(audio_emb, opt.n_poses, window_step)
        motions_list = get_windows(motions, opt.n_poses, window_step)
        
        # Add HuBERT features (same as official runner)
        add_cond = {}
        print("[OFFICIAL EXPORT] Extracting HuBERT features...")
        add_cond["pretrain_aud_feat"] = get_hubert_from_16k_speech_long(
            hubert_model, wav2vec2_processor, 
            torch.from_numpy(aud_ori).unsqueeze(0).to(opt.device), 
            device=opt.device
        )
        add_cond["pretrain_aud_feat"] = F.interpolate(
            add_cond["pretrain_aud_feat"].swapaxes(-1,-2).unsqueeze(0), 
            size=audio_emb.shape[-2], 
            mode='linear', 
            align_corners=True
        ).swapaxes(-1,-2)
        
        # Put dict values into device
        if isinstance(add_cond, dict):
            for key in add_cond.keys():
                add_cond[key] = add_cond[key].to(opt.device)
        
        if add_cond not in [None, {}]:
            add_cond_list = get_windows(add_cond, opt.n_poses, window_step)
        
        # Generate motion for each window (same as official runner)
        out_motions = []
        print(f"[OFFICIAL EXPORT] Generating motion for {len(audio_emb_list)} windows...")
        for ii, [audio_emb_win, motions_win] in enumerate(zip(audio_emb_list, motions_list)):
            print(f"[OFFICIAL EXPORT] Window {ii+1}/{len(audio_emb_list)}")
            
            if add_cond not in [None, {}]:
                add_cond_win = add_cond_list[ii]
            
            inpaint_dict = {}
            if opt.overlap_len > 0:
                inpaint_dict['gt'] = torch.zeros_like(motions_win)
                inpaint_dict['outpainting_mask'] = torch.zeros_like(
                    motions_win, dtype=torch.bool, device=motions_win.device
                )
                
                if ii == 0:
                    if opt.fix_very_first:
                        inpaint_dict['outpainting_mask'][..., :opt.overlap_len, :] = True
                        inpaint_dict['gt'][:, :opt.overlap_len, ...] = motions_win[:, -opt.overlap_len:, ...]
                    else:
                        pass
                elif ii > 0:
                    inpaint_dict['outpainting_mask'][..., :opt.overlap_len, :] = True
                    inpaint_dict['gt'][:, :opt.overlap_len, ...] = outputs[:, -opt.overlap_len:, ...]
            
            outputs = trainer.generate_batch(
                audio_emb_win, p_id_tensor, opt.net_dim_pose, add_cond_win, inpaint_dict
            )
            
            outputs_np = outputs.cpu().numpy()
            if ii == len(motions_list) - 1:
                out_motions.append(outputs_np)
            else:
                out_motions.append(outputs_np[:, :window_step])
        
        # Concatenate all windows
        out_motions = np.concatenate(out_motions, 1)
        print(f"[OFFICIAL EXPORT] Total frames: {out_motions.shape[1]}")
    else:
        # Use pre-generated motion
        print("[OFFICIAL EXPORT] Using pre-generated motion from wrapper...")
        out_motions = pre_generated_motion.cpu().numpy()
        print(f"[OFFICIAL EXPORT] Pre-generated motion shape: {out_motions.shape}")
    
    # Split gesture and expression (same as official runner)
    if opt.unidiffuser or opt.net_dim_pose == 192:
        out_motions, out_expression = np.split(out_motions, [opt.split_pos], axis=-1)
    
    # Handle axis-angle to Euler conversion (same as official runner)
    if opt.axis_angle:
        # Load normalization stats
        mean_axis_path = f"data/BEAT/beat_cache/{opt.beat_cache_name}/train/axis_angle_mean.npy"
        std_axis_path = f"data/BEAT/beat_cache/{opt.beat_cache_name}/train/axis_angle_std.npy"
        mean_euler_path = f"data/BEAT/beat_cache/{opt.beat_cache_name}/train/bvh_rot/bvh_mean.npy"
        std_euler_path = f"data/BEAT/beat_cache/{opt.beat_cache_name}/train/bvh_rot/bvh_std.npy"
        
        test_dataset_mean_axis = torch.from_numpy(np.load(mean_axis_path)).float()
        test_dataset_std_axis = torch.from_numpy(np.load(std_axis_path)).float()
        test_dataset_mean_pose = torch.from_numpy(np.load(mean_euler_path)).float()
        test_dataset_std_pose = torch.from_numpy(np.load(std_euler_path)).float()
        
        # Convert axis-angle to Euler (same as official runner)
        out_motions = torch.from_numpy(out_motions)
        denorm_out = out_motions * test_dataset_std_axis + test_dataset_mean_axis
        B, T, C = denorm_out.shape
        euler_out = rot_cvt.axis_angle_to_euler_angles(denorm_out.reshape(B, T, C//3, 3)).reshape(B, T, C)
        euler_out = euler_out * (180 / np.pi)
        out_motions = (euler_out - test_dataset_mean_pose) / test_dataset_std_pose
        out_motions = out_motions.numpy()
    
    # Save outputs (same as official runner)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create gesture and expression directories
    gesture_dir = os.path.join(output_dir, 'gesture')
    expression_dir = os.path.join(output_dir, 'expression')
    os.makedirs(os.path.join(gesture_dir, 'bvh'), exist_ok=True)
    os.makedirs(os.path.join(expression_dir, 'face_json'), exist_ok=True)
    
    # Save gesture as numpy and BVH
    np.save(os.path.join(gesture_dir, f"{name.split('.')[0]}.npy"), out_motions)
    out_denorm_euler = euler_out.reshape(-1, opt.dim_pose).numpy()
    trainer.result2target_vis(
        out_denorm_euler, 
        os.path.join(gesture_dir, 'bvh'), 
        f"{name.split('.')[0]}.bvh"
    )
    print(f"[OFFICIAL EXPORT] Saved BVH to {gesture_dir}/bvh/{name.split('.')[0]}.bvh")
    
    # Save expression as numpy and JSON
    if opt.unidiffuser or opt.net_dim_pose == 192:
        np.save(os.path.join(expression_dir, f"{name.split('.')[0]}.npy"), out_expression)
        trainer.write_face_json(
            out_expression, 
            os.path.join(expression_dir, 'face_json', f"{name.split('.')[0]}.json")
        )
        print(f"[OFFICIAL EXPORT] Saved JSON to {expression_dir}/face_json/{name.split('.')[0]}.json")
    
    print("[OFFICIAL EXPORT] Export complete!")
    return gesture_dir, expression_dir


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
    
    # Set test mode before loading checkpoint (mirrors runner.py test methods)
    opt.is_train = False
    
    # Load trained weights (mirror runner.py)
    opt.save_root = os.path.join(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = os.path.join(opt.save_root, 'model')
    ckpt_path = os.path.join(opt.model_dir, opt.ckpt)
    print(f"Loading checkpoint: {ckpt_path}")
    trainer.load(ckpt_path)
    trainer.encoder.eval()
    
    # Enable HuBERT conditioning to match checkpoint training config
    # The checkpoint name 'beat_GesExpr_unify_addHubert_encodeHubert_mlpIncludeX_condRes_LN' indicates HuBERT was used
    print("Enabling HuBERT conditioning (checkpoint was trained with addHubert)")
    opt.addHubert = True
    
    # Initialize wrapper
    print("Initializing realtime wrapper...")
    
    # Enable non-streaming sanity check mode
    DiffSHEGRealtimeWrapper.NON_STREAMING_SANITY_CHECK = True
    
    print("="*60)
    print(f"{'ENABLING' if DiffSHEGRealtimeWrapper.NON_STREAMING_SANITY_CHECK else 'DISABLING'} NON_STREAMING_SANITY_CHECK MODE")
    print("This will wait for full audio before generating using official pipeline")
    print("="*60)

    # Waypoint collector - stores waypoints as they're generated
    collected_waypoints = []
    
    def waypoint_handler(waypoint):
        """Collect waypoints generated by the wrapper"""
        collected_waypoints.append({
            'index': waypoint.waypoint_index,
            'timestamp': waypoint.timestamp,
            'gesture_data': waypoint.gesture_data.copy()
        })

    wrapper = DiffSHEGRealtimeWrapper(
        diffsheg_model=trainer,
        opt=opt,
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
    if DiffSHEGRealtimeWrapper.NON_STREAMING_SANITY_CHECK:
        print("\n[SANITY CHECK MODE] Waiting for audio completion and generation...")
        # Wait for sanity check to complete
        wrapper.sanity_check_generation_complete.wait()
        print("[SANITY CHECK MODE] Generation complete!")
    else:
        print("\nWaiting for gesture generation and playback to complete...")
        # Wait for audio duration + cleanup timeout
        wait_time = audio_duration + wrapper.cleanup_timeout + 1.0
        print(f"Waiting {wait_time:.1f}s...")
        time.sleep(wait_time)
    
    # Stop wrapper
    print("\nStopping wrapper...")
    wrapper.stop()
    
    # Export waypoints to visualization format using official trainer methods
    print("\n" + "="*60)
    print("Exporting gestures to visualization format...")
    print("="*60)
    
    output_dir = 'results/realtime_test'
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the generated motion stored in wrapper (avoids regeneration)
    if hasattr(wrapper, 'last_generated_motion') and wrapper.last_generated_motion is not None:
        out_motions = wrapper.last_generated_motion.cpu().numpy()  # [1, T, 192]
        print(f"[EXPORT] Using generated motion from wrapper: shape={out_motions.shape}")
    else:
        # Fallback: reconstruct from collected waypoints
        print(f"[EXPORT] Reconstructing motion from {len(collected_waypoints)} waypoints...")
        collected_waypoints.sort(key=lambda x: x['index'])
        waypoint_data = [wp['gesture_data'] for wp in collected_waypoints]
        out_motions = np.stack(waypoint_data, axis=0)  # [T, 192]
        out_motions = np.expand_dims(out_motions, axis=0)  # [1, T, 192]
    
    # Split gesture and expression
    out_gestures, out_expression = np.split(out_motions, [opt.split_pos], axis=-1)
    
    # Create output directories
    gesture_dir = os.path.join(output_dir, 'gesture')
    expression_dir = os.path.join(output_dir, 'expression')
    os.makedirs(os.path.join(gesture_dir, 'bvh'), exist_ok=True)
    os.makedirs(os.path.join(expression_dir, 'face_json'), exist_ok=True)
    
    # Load normalization statistics (axis-angle for gestures)
    from datasets import BeatDataset
    test_dataset = BeatDataset(opt, "test")
    
    # Convert axis-angle to Euler and save
    audio_name = os.path.basename(audio_path).split('.')[0]
    
    # Convert gestures: axis-angle -> Euler
    out_gestures_tensor = torch.from_numpy(out_gestures)
    denorm_out = out_gestures_tensor * test_dataset.std_pose_axis_angle + test_dataset.mean_pose_axis_angle
    B, T, C = denorm_out.shape
    euler_out = rot_cvt.axis_angle_to_euler_angles(denorm_out.reshape(B, T, C//3, 3)).reshape(B,T,C)
    euler_out = euler_out * (180 / np.pi)  # Convert to degrees
    out_gestures = (euler_out - test_dataset.mean_pose) / test_dataset.std_pose
    out_gestures = out_gestures.numpy()
    
    # Save gesture as BVH
    np.save(os.path.join(gesture_dir, f"{audio_name}.npy"), out_gestures)
    out_denorm_euler = euler_out.reshape(-1, opt.dim_pose).numpy()
    trainer.result2target_vis(out_denorm_euler, os.path.join(gesture_dir, 'bvh'), f"{audio_name}.bvh")
    
    # Save expression as JSON
    np.save(os.path.join(expression_dir, f"{audio_name}.npy"), out_expression)
    trainer.write_face_json(out_expression, os.path.join(expression_dir, 'face_json', f"{audio_name}.json"))
    
    # Copy audio
    audio_output_path = os.path.join(output_dir, os.path.basename(audio_path))
    shutil.copy(audio_path, audio_output_path)
    print(f"[EXPORT] Saved gesture, expression, and audio to {output_dir}")
    
    # Prepare Blender paths
    bvh_path = os.path.join(gesture_dir, 'bvh', f'{audio_name}.bvh')
    json_path = os.path.join(expression_dir, 'face_json', f'{audio_name}.json')
    video_path = os.path.join(output_dir, f'{audio_name}.mp4')
    
    print("\n" + "="*60)
    print("Next Steps for Blender Visualization:")
    print("="*60)
    print("1. Open assets/beat_visualize.blend in Blender")
    print("2. Open the Scripting tab and update these paths in blender_script_text:")
    print(f"\n   FACE_ANIM_NAME = r\"{os.path.abspath(json_path)}\"")
    print(f"   SOUND_NAME = r\"{os.path.abspath(audio_output_path)}\"")
    print(f"   MOTION_NAME = r\"{os.path.abspath(bvh_path)}\"")
    print(f"   VIDEO_NAME = r\"{os.path.abspath(video_path)}\"")
    print("\n3. Run the script in Blender (Alt+P) to render the video")
    print("="*60)
    
    print("\nTest complete!")


if __name__ == "__main__":
    main()
