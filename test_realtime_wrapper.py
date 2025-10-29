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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from options.train_options import TrainCompOptions
from models import MotionTransformer, UniDiffuser
from trainers import DDPMTrainer_beat
from diffsheg_realtime_wrapper import DiffSHEGRealtimeWrapper


def waypoint_handler(waypoint):
    """
    Callback function to handle waypoint execution.
    
    Args:
        waypoint: GestureWaypoint object containing waypoint information
    """
    print(f"[WAYPOINT] Received waypoint {waypoint.waypoint_index} at timestamp {waypoint.timestamp:.3f}s - gesture_data shape: {waypoint.gesture_data.shape}")


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
    
    # Initialize trainer (which handles model loading)
    print("Initializing trainer...")
    trainer = DDPMTrainer_beat(opt, model, eval_model=None)
    
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
    
    print("\nTest complete!")


if __name__ == "__main__":
    main()
