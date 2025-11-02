"""
DiffSHEG Realtime Wrapper for GPT-4o Dialogue System Integration

This wrapper class manages the integration of DiffSHEG gesture generation with
a real-time dialogue system that receives audio chunks from GPT-4o-realtime.

OPERATING MODES:
================

The wrapper supports two operating modes via the NON_STREAMING_SANITY_CHECK flag:

1. STREAMING MODE (default):
   - NON_STREAMING_SANITY_CHECK = False
   - Real-time gesture generation as audio chunks arrive
   - Uses incremental windowing with overlap context
   - Audio is truncated to [0, window_end] for each window generation
   
2. NON-STREAMING SANITY CHECK MODE:
   - NON_STREAMING_SANITY_CHECK = True
   - Waits for complete audio before generating
   - Generates window by window with audio truncation
   - Useful for debugging and comparing with official offline generation

Both modes use the EXACT same official generation pipeline from DiffSHEG:
- Official mel spectrogram extraction (librosa.feature.melspectrogram)
- Official HuBERT feature extraction (get_hubert_from_16k_speech_long)
- Official trainer.generate_batch() method with proper inpainting

To enable sanity check mode, set the flag BEFORE creating the wrapper:
    DiffSHEGRealtimeWrapper.NON_STREAMING_SANITY_CHECK = True
    wrapper = DiffSHEGRealtimeWrapper(...)

DEBUGGING FEATURES:
===================

SAVE_WINDOWS flag:
- When enabled, saves each audio window sent to gesture generation as a WAV file
- Files are saved to: embodiment_manager/logger/logs/audio_windows_<timestamp>/
- Filename format: utt<utterance_id>_win<window_index>_samples<start>-<end>.wav
- Useful for comparing audio content when troubleshooting performance issues
- To enable:
    DiffSHEGRealtimeWrapper.SAVE_WINDOWS = True
    wrapper = DiffSHEGRealtimeWrapper(...)

USE_TEST_DATA_FOR_WINDOW flag:
- When enabled, pre-loads the first window from a test audio file and uses it for ALL generations
- Instead of using incoming audio chunks, always generates gestures from the same test window
- Useful for isolating performance issues: if generation is fast with test data but slow with
  real data, the problem is likely with the audio data format/content, not the generation pipeline
- Test audio is loaded from: ../floor_coordinator/test_audio/test_audio.wav
- To enable:
    DiffSHEGRealtimeWrapper.USE_TEST_DATA_FOR_WINDOW = True
    wrapper = DiffSHEGRealtimeWrapper(...)
"""

import threading
import time
import queue
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
from datetime import datetime
import hashlib
import gc

# Add parent directory to path to import logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger.logger import setup_logger
import wave
from copy import deepcopy

try:
    from trainers.ddpm_beat_trainer import get_hubert_from_16k_speech_long
except ImportError:
    get_hubert_from_16k_speech_long = None


ENABLE_CLEARING_CONTENT= True
ENABLE_CLEARING_META1 = True
ENAGLE_CLEARING_META2 = True

@dataclass
class GestureWaypoint:
    """
    Represents a single gesture frame at a specific timestamp.
    
    Each waypoint contains the gesture data for ONE frame (not a window).
    """
    waypoint_index: int  # Sequential frame index (0, 1, 2, ...)
    timestamp: float  # Time in seconds from utterance start when this frame should be executed
    gesture_data: np.ndarray  # Gesture pose vector of shape (C,) for this single frame
    is_for_execution: bool = True  # True if this frame should be executed, False if only for inpainting context


@dataclass
class WaypointWindow:
    """
    Represents one window generation containing multiple waypoints.
    
    For BEAT with window_size=34, overlap_len=4, window_step=30:
    - execution_waypoints: First 30 waypoints (frames 0-29) - executed and exported
    - context_waypoints: Last 4 waypoints (frames 30-33) - used for next window's inpainting
    
    The window generates 34 total waypoints but only the first 30 are for execution.
    """
    window_index: int  # Sequential window index (0, 1, 2, ...)
    execution_waypoints: List[GestureWaypoint]  # Waypoints for execution (window_step frames)
    context_waypoints: List[GestureWaypoint]  # Waypoints for next window's inpainting (overlap_len frames)




class Utterance:
    """Tracks an ongoing or completed utterance with audio stored as concatenated samples."""
    
    PLACE_HOLDER_ID = -1
    PLACE_HOLDER_TIMESTAMP = -1

    def __init__(
        self, 
        utterance_id: int,
        sample_rate: int,
        gesture_fps: int,
        window_size: int,
        window_step: int
    ):
        self.utterance_id = utterance_id
        self.start_time: Optional[float] = Utterance.PLACE_HOLDER_TIMESTAMP
        self.last_chunk_received_time: float = Utterance.PLACE_HOLDER_TIMESTAMP # Track when last chunk arrived
        
        # Audio storage: concatenated samples instead of chunks
        self.sample_rate = sample_rate
        self.audio_samples: bytes = b''  # Concatenated raw audio samples (s16le encoded)
        self.bytes_per_sample = 2  # s16le encoding uses 2 bytes per sample
        
        # Generation state: tracks which audio window to generate next (in terms of sample indices)
        self.gesture_fps = gesture_fps
        self.window_size = window_size  # Number of frames per window
        self.window_step = window_step  # Number of non-overlapping frames per window
        
        # Initialize window indices - start from sample 0 (beginning of utterance)
        window_duration_samples = int((window_size / gesture_fps) * sample_rate)
        
        self.next_window_start_sample: int = 0
        self.next_window_end_sample: int = window_duration_samples
        
        # Gesture data structures
        self.windows: List[WaypointWindow] = []  # All generated windows
        self.execution_waypoints: List[GestureWaypoint] = []  # Flat list of waypoints for execution (only is_for_execution=True)
        self.last_executed_waypoint_index: int = -1  # Cursor for playback
        self.waypoints_lock = threading.Lock()
    
    def add_audio_samples(self, audio_data):
        """
        Add audio samples to the utterance.
        
        Args:
            audio_data: Audio data as list of integers
        """
        if isinstance(audio_data, list):
            # Convert list of integers to bytes (as in process_system_reference_audio)
            audio_bytes = bytes(audio_data)
        else:
            raise TypeError(f"Unsupported audio_data type: {type(audio_data)}")
        

        self.audio_samples += audio_bytes
    
    def get_total_samples(self) -> int:
        """Get total number of audio samples accumulated."""
        return len(self.audio_samples) // self.bytes_per_sample
    
    def get_audio_window(self, start_sample: int, end_sample: int) -> bytes:
        """
        Extract raw audio bytes for a specific window.
        
        Args:
            start_sample: Starting sample index
            end_sample: Ending sample index (exclusive)
            
        Returns:
            Raw audio bytes (s16le encoded). Caller is responsible for format conversion.
        """
        start_byte = start_sample * self.bytes_per_sample
        end_byte = end_sample * self.bytes_per_sample
        
        if len(self.audio_samples) < end_byte:
            raise RuntimeError('Insufficient audio bytes!')
        
        
        if start_byte >= len(self.audio_samples):
            return b''
        
        return self.audio_samples[start_byte:end_byte]
    
    def update_window_indices(self):
        """
        Update window indices for the next generation window.
        Advances by window_step frames worth of samples.
        """
        step_duration_samples = int((self.window_step / self.gesture_fps) * self.sample_rate)
        window_duration_samples = int((self.window_size / self.gesture_fps) * self.sample_rate)
        
        self.next_window_start_sample += step_duration_samples
        self.next_window_end_sample = self.next_window_start_sample + window_duration_samples
    
    def add_window(self, window: WaypointWindow):
        """Add a generated window and extract execution waypoints."""
        with self.waypoints_lock:
            self.windows.append(window)
            # Add all execution waypoints to the flat list for playback
            self.execution_waypoints.extend(window.execution_waypoints)
    
    def clear(self):
        """
        Clear all content of the utterance for reuse.
        This resets all data structures while keeping the object alive.
        """
        
        with self.waypoints_lock:
            
            if ENABLE_CLEARING_META1:
                self.utterance_id = Utterance.PLACE_HOLDER_ID  # Reset to placeholder ID

                # Reset window indices to start from sample 0
                window_duration_samples = int((self.window_size / self.gesture_fps) * self.sample_rate)


                self.next_window_start_sample = 0
                self.next_window_end_sample = window_duration_samples

                
            if ENAGLE_CLEARING_META2:

                # Clear timing information
                # Note: Setting to None can trigger GC, but we've disabled GC
                # during critical CUDA operations to prevent interference
                self.start_time = Utterance.PLACE_HOLDER_TIMESTAMP
                self.last_chunk_received_time = Utterance.PLACE_HOLDER_TIMESTAMP

            if ENABLE_CLEARING_CONTENT:
                # Clear audio data
                self.audio_samples = b''
                
                # Clear gesture data structures
                for window in self.windows:
                    for wp in window.execution_waypoints:
                        del wp.gesture_data
                    for wp in window.context_waypoints:
                        del wp.gesture_data
                
                for wp in self.execution_waypoints:
                    del wp.gesture_data
                self.windows = []  # Create new list instead of .clear()
                self.execution_waypoints = []  # Create new list
                self.last_executed_waypoint_index = -1


    def get_waypoint_for_interval(self, current_time: float, interval_duration: float = 0.01) -> Optional[GestureWaypoint]:
        """
        Find the waypoint that should be executed in the next interval.
        
        Args:
            current_time: Current playback time in seconds from utterance start
            interval_duration: Duration of the upcoming interval (default 10ms = 0.01s)
            
        Returns:
            The waypoint to execute, or None if no waypoint falls in this interval.
        """
        with self.waypoints_lock:
            if not self.execution_waypoints:
                return None
            
            interval_end = current_time + interval_duration
            
            # Search from the last executed waypoint onwards
            search_start = self.last_executed_waypoint_index + 1
            
            for i in range(search_start, len(self.execution_waypoints)):
                waypoint = self.execution_waypoints[i]
                
                # Check if this waypoint falls within the upcoming interval
                if current_time <= waypoint.timestamp < interval_end:
                    self.last_executed_waypoint_index = i
                    return waypoint
                
                # If waypoint is beyond the interval, we can stop searching
                if waypoint.timestamp >= interval_end:
                    break
            
            return None



class DiffSHEGRealtimeWrapper:
    """
    Wrapper for integrating DiffSHEG with real-time dialogue systems.
    
    Features:
    - Tracks utterance lifecycle and audio chunks
    - Manages two threads: playback monitoring and gesture generation
    - Generation thread uses snapshot-based approach for lock-free inference
    
    NON_STREAMING_SANITY_CHECK mode:
    - When enabled, waits for full audio before generating
    - Useful for debugging and comparing with official offline generation
    - When False (default), operates in streaming mode for real-time interaction
    
    USE_CONSTRAINED_FEATURES mode:
    - When False (default), streaming mode computes features from [0, window_end]
    - When True, constrains audio context to AUDIO_DUR_FOR_FEATURES seconds
    - This ensures stable performance regardless of utterance length
    """
    
    # Global flag for sanity check mode
    NON_STREAMING_SANITY_CHECK = False
    
    # Global flags for constrained feature extraction
    USE_CONSTRAINED_FEATURES = True
    AUDIO_DUR_FOR_FEATURES = 5.0  # Duration in seconds for constrained audio context
    
    # Global flag for saving audio windows for debugging
    SAVE_WINDOWS = False


    # Global flag for using pre-loaded test data for debugging
    USE_TEST_DATA_FOR_WINDOW = False

    
    def __init__(
        self,
        diffsheg_model,
        opt,
        config: dict = None,
        audio_sr: int = None,
        device: str = None,
        cleanup_timeout: float = None,
        waypoint_callback = None
    ):
        """
        Initialize the wrapper.
        
        Args:
            diffsheg_model: The DiffSHEG trainer instance (DDPMTrainer_beat)
            opt: Configuration options
            config: Configuration dictionary (typically from YAML). If provided, will be used
                   for default values. Individual parameters override config values.
            audio_sr: Audio sample rate. If None, reads from config or uses 16000.
            device: Computing device. If None, reads from config or uses opt.device.
            cleanup_timeout: Seconds to wait after playback ends before auto-cleanup.
                           If None, reads from config or uses 2.0.
            waypoint_callback: Optional callback function to execute waypoints.
                             Should accept a GestureWaypoint object as parameter.
                             If None, waypoints are generated but not executed.
        """
        self.model = diffsheg_model
        self.opt = opt
        self.config = config or {}
        
        # Get co_speech_gestures config section
        gesture_config = self.config.get('co_speech_gestures', {})
        
        # Load parameters with priority: explicit parameter > config > default
        self.audio_sr = (
            audio_sr if audio_sr is not None
            else gesture_config.get('audio_sr', 16000)
        )
        
        # Handle device - ensure it's a torch.device object
        device_str = (
            device if device is not None
            else gesture_config.get('gpu_device', str(opt.device))
        )
        # Convert string to torch.device if necessary
        if isinstance(device_str, str):
            self.device = torch.device(device_str)
        else:
            self.device = device_str
            

        self.cleanup_timeout = (
            cleanup_timeout if cleanup_timeout is not None
            else gesture_config.get('cleanup_timeout', 2.0)
        )
            
        # Store waypoint callback
        self.waypoint_callback = waypoint_callback
        
        # Initialize logger
        self.logger = setup_logger(
            logger_name='diffsheg_realtime_wrapper',
            file_log_level="DEBUG",
            terminal_log_level="INFO"
        )
        self.logger.info("DiffSHEG Realtime Wrapper initialized")
        self.logger.info(f"Configuration: sample_rate={self.audio_sr}, device={self.device}")
        
        # Setup audio window save directory if SAVE_WINDOWS is enabled
        self.audio_windows_dir = None
        if self.SAVE_WINDOWS:
            # Create directory beside the log files
            log_root = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'logger',
                'logs'
            )
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.audio_windows_dir = os.path.join(log_root, f'audio_windows_{timestamp}')
            os.makedirs(self.audio_windows_dir, exist_ok=True)
            self.logger.info(f"SAVE_WINDOWS enabled: saving audio windows to {self.audio_windows_dir}")
            self.window_save_counter = 0  # Counter for saved windows
        
        # Check if HuBERT features are needed
        self.use_hubert = getattr(opt, 'addHubert', False) or getattr(opt, 'expAddHubert', False)
        if self.use_hubert:
            self.logger.info("HuBERT features enabled - loading models...")
            from transformers import Wav2Vec2Processor, HubertModel
            self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
            self.hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
            self.hubert_model = self.hubert_model.to(self.device)
            self.hubert_model.eval()
            self.logger.info("HuBERT models loaded successfully")
        else:
            self.wav2vec2_processor = None
            self.hubert_model = None
            self.logger.info("HuBERT features disabled")
        
        # DiffSHEG configuration (needed before creating utterance)
        self.window_size = opt.n_poses  # e.g., 34 frames
        self.overlap_len = opt.overlap_len  # e.g., 4 frames
        self.window_step = self.window_size - self.overlap_len
        
        # Compute frames per audio second
        # DiffSHEG uses 15 FPS for BEAT dataset
        self.gesture_fps = 15
        
        # Current utterance tracking (created once and reused)
        self.current_utterance: Utterance = Utterance(
            utterance_id=Utterance.PLACE_HOLDER_ID,  # Placeholder ID, will be updated when first chunk arrives
            sample_rate=self.audio_sr,
            gesture_fps=self.gesture_fps,
            window_size=self.window_size,
            window_step=self.window_step
        )
        self.utterance_lock = threading.Lock()
        self.logger.info("Persistent utterance object created (will be reused)")
        
        # Track stopped/timed-out utterances to reject late-arriving chunks
        self.stopped_utterances: set = set()  # Set of utterance_ids that have been stopped or timed out
        
        # Threading
        self.running = False
        self.playback_managing_thread: Optional[threading.Thread] = None
        self.generation_thread: Optional[threading.Thread] = None
        self.sanity_check_thread: Optional[threading.Thread] = None
        
        # Sanity check mode state
        self.sanity_check_audio_complete = threading.Event()
        self.sanity_check_generation_complete = threading.Event()
        
        # Store last generated motion (for avoiding regeneration during export)
        self.last_generated_motion = None
        
        self.logger.info(f"DiffSHEG parameters: window_size={self.window_size}, overlap={self.overlap_len}, step={self.window_step}, fps={self.gesture_fps}")

        # Warm up CUDA context with dummy inference
        self.logger.info("Warming up CUDA context...")
        self._warmup_cuda_context()
        self.logger.info("CUDA context warmed up successfully")

        # Load test data for debugging if enabled
        self.test_audio_window = None
        if self.USE_TEST_DATA_FOR_WINDOW:
            self.logger.info("USE_TEST_DATA_FOR_WINDOW enabled - loading test audio...")
            self._load_test_audio_window()

        # Audio feature extraction parameters (EXACTLY matching official test_custom_aud)
        # Official code: mel = librosa.feature.melspectrogram(y=aud, sr=18000, hop_length=1200, n_mels=128)
        # NO explicit n_fft, win_length, window, center, or power_to_db conversion!
        self.mel_sample_rate = 18000
        self.mel_hop_length = 1200
        self.mel_num_mels = 128
        mel_frame_rate = self.mel_sample_rate / self.mel_hop_length
        if abs(mel_frame_rate - self.gesture_fps) > 1e-3:
            self.logger.warning(
                f"Mel frame rate {mel_frame_rate:.3f} does not match gesture FPS {self.gesture_fps}; check configuration."
            )

    def _warmup_cuda_context(self):
        """
        Warm up CUDA context with dummy inference to eliminate cold-start overhead.
        """
        try:
            with torch.no_grad():
                # Warm up main model
                dummy_audio = torch.randn(1, self.window_size, 128, device=self.device)
                dummy_pid = torch.zeros((1, 1), dtype=torch.long, device=self.device)
                dummy_pid_onehot = self.model.one_hot(dummy_pid, self.opt.speaker_dim)
                dummy_add_cond = {}
                
                if self.use_hubert:
                    # CRITICAL: Warm up HuBERT model
                    self.logger.debug("Warming up HuBERT model...")
                    dummy_audio_samples = torch.randn(1, 16000, device=self.device)  # 1 second of audio
                    dummy_hubert_feat = get_hubert_from_16k_speech_long(
                        self.hubert_model,
                        self.wav2vec2_processor,
                        dummy_audio_samples,
                        device=self.device,
                        return_on_cpu=False
                    )
                    # Interpolate to match window size
                    dummy_hubert_feat = F.interpolate(
                        dummy_hubert_feat.swapaxes(-1,-2).unsqueeze(0),
                        size=self.window_size,
                        mode='linear',
                        align_corners=True
                    ).swapaxes(-1,-2)
                    dummy_add_cond['pretrain_aud_feat'] = dummy_hubert_feat
                    self.logger.debug(f"HuBERT warm-up complete, output shape: {dummy_hubert_feat.shape}")
                
                dummy_inpaint_dict = {}
                
                # Run inference
                self.logger.debug("Running warm-up inference...")
                _ = self.model.generate_batch(
                    dummy_audio, 
                    dummy_pid_onehot, 
                    self.opt.net_dim_pose,
                    dummy_add_cond,
                    dummy_inpaint_dict
                )
                
                # Synchronize to ensure all operations complete
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                
                self.logger.debug("Warm-up inference completed")
                
        except Exception as e:
            self.logger.warning(f"CUDA warm-up failed (non-critical): {e}")

    def _load_test_audio_window(self):
        """
        Load a window's worth of test audio data for debugging.
        This loads the first window from a test audio file and stores it for repeated use.
        """
        try:
            import librosa
            
            # Try to find test audio file
            test_audio_path = None
            possible_paths = [
                # Try relative to this file
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    'floor_coordinator', 'test_audio', 'test_audio.wav'
                ),
                # Try relative to embodiment_manager
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    '..', 'floor_coordinator', 'test_audio', 'test_audio.wav'
                ),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    test_audio_path = path
                    break
            
            if test_audio_path is None:
                self.logger.error("Could not find test audio file. USE_TEST_DATA_FOR_WINDOW will not work!")
                self.test_audio_window = None
                return
            
            self.logger.info(f"Loading test audio from: {test_audio_path}")
            
            # Load audio with librosa
            audio, sr = librosa.load(test_audio_path, sr=None, mono=True)
            
            # Resample to target sample rate if needed
            if sr != self.audio_sr:
                self.logger.info(f"Resampling test audio from {sr} Hz to {self.audio_sr} Hz")
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.audio_sr)
            
            # Calculate how many samples we need for one window
            window_duration_samples = int((self.window_size / self.gesture_fps) * self.audio_sr)
            
            # Extract first window
            if len(audio) < window_duration_samples:
                self.logger.warning(f"Test audio is too short ({len(audio)} samples), padding to {window_duration_samples}")
                audio = np.pad(audio, (0, window_duration_samples - len(audio)), mode='constant')
            
            audio_window = audio[:window_duration_samples]
            
            # Convert to int16 and then to bytes (s16le format)
            audio_int16 = (audio_window * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            # Store the test audio window
            self.test_audio_window = audio_bytes
            
            self.logger.info(
                f"Test audio window loaded successfully: "
                f"{len(audio_bytes)} bytes, {window_duration_samples} samples, "
                f"{window_duration_samples / self.audio_sr:.3f} seconds"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load test audio window: {e}")
            self.logger.exception(e)
            self.test_audio_window = None

    def _save_audio_window(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        utterance_id: int,
        window_index: int,
        window_start_sample: int,
        window_end_sample: int
    ):
        """
        Save an audio window to a WAV file and text document for debugging.
        
        Saves two files:
        1. WAV file: playable audio for listening
        2. TXT file: raw audio samples as list of integers (byte values 0-255)
        
        Args:
            audio_bytes: Raw audio bytes (s16le encoded)
            sample_rate: Audio sample rate (e.g., 16000)
            utterance_id: ID of the utterance
            window_index: Index of the window
            window_start_sample: Starting sample index of the window
            window_end_sample: Ending sample index of the window
        """
        if not self.SAVE_WINDOWS or self.audio_windows_dir is None:
            return
        
        try:
            # Create base filename with detailed information
            base_filename = f"utt{utterance_id:03d}_win{window_index:04d}_samples{window_start_sample}-{window_end_sample}"
            
            # ===== Save WAV file =====
            # Convert bytes to numpy array (s16le format)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Convert to float32 in range [-1, 1] for soundfile
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            wav_filepath = os.path.join(self.audio_windows_dir, base_filename + ".wav")
            sf.write(wav_filepath, audio_float, sample_rate)
            
            # ===== Save TXT file with audio samples as list of integers =====
            # Convert bytes to list of integers (byte values 0-255)
            # This matches the format used in the rest of the code
            audio_as_int_list = list(audio_bytes)
            
            txt_filepath = os.path.join(self.audio_windows_dir, base_filename + ".txt")
            with open(txt_filepath, 'w') as f:
                f.write(f"# Audio Window: Utterance {utterance_id}, Window {window_index}\n")
                f.write(f"# Sample range: {window_start_sample} - {window_end_sample}\n")
                f.write(f"# Sample rate: {sample_rate} Hz\n")
                f.write(f"# Encoding: s16le (2 bytes per sample)\n")
                f.write(f"# Total bytes: {len(audio_bytes)}\n")
                f.write(f"# Total samples: {len(audio_array)}\n")
                f.write(f"# Duration: {len(audio_array)/sample_rate:.4f} seconds\n")
                f.write(f"#\n")
                f.write(f"# Audio bytes as list of integers (0-255):\n")
                f.write(f"audio_data = {audio_as_int_list}\n")
                f.write(f"\n")
                f.write(f"# Statistics:\n")
                f.write(f"# Min byte value: {min(audio_as_int_list) if audio_as_int_list else 'N/A'}\n")
                f.write(f"# Max byte value: {max(audio_as_int_list) if audio_as_int_list else 'N/A'}\n")
                f.write(f"# First 20 bytes: {audio_as_int_list[:20]}\n")
                f.write(f"# Last 20 bytes: {audio_as_int_list[-20:]}\n")
            
            self.window_save_counter += 1
            if self.window_save_counter % 10 == 0:
                self.logger.debug(f"Saved {self.window_save_counter} audio windows (WAV + TXT) to {self.audio_windows_dir}")
                
        except Exception as e:
            self.logger.error(f"Failed to save audio window: {e}")

    def start(self):
        """Start the wrapper threads."""
        self.running = True
        
        # Log mode information
        if DiffSHEGRealtimeWrapper.NON_STREAMING_SANITY_CHECK:
            self.logger.info("="*60)
            self.logger.info("SANITY CHECK MODE: NON-STREAMING")
            self.logger.info("Using wrapper's own generation pipeline")
            self.logger.info("="*60)
        else:
            self.logger.info("="*60)
            self.logger.info("STREAMING MODE: Real-time generation")
            if DiffSHEGRealtimeWrapper.USE_CONSTRAINED_FEATURES:
                self.logger.info(f"CONSTRAINED FEATURES: Enabled (duration={DiffSHEGRealtimeWrapper.AUDIO_DUR_FOR_FEATURES}s)")
            else:
                self.logger.info("CONSTRAINED FEATURES: Disabled (using full audio context)")
            self.logger.info("="*60)
        
        if self.NON_STREAMING_SANITY_CHECK:
            # Sanity check mode: monitor audio completion, then generate all at once
            self.sanity_check_thread = threading.Thread(target=self._sanity_check_loop, daemon=True)
            self.sanity_check_thread.start()
            self.logger.info("NON_STREAMING_SANITY_CHECK mode enabled - waiting for full audio before generation")
        else:
            # Normal streaming mode
            self.playback_managing_thread = threading.Thread(target=self._playback_managing_loop, daemon=True)
            self.generation_thread = threading.Thread(target=self._generation_loop, daemon=True)
            self.playback_managing_thread.start()
            self.generation_thread.start()
            self.logger.info("Playback managing and generation threads started")
        
    def stop(self):
        """Stop the wrapper threads."""
        self.logger.info("Stopping wrapper threads...")
        self.running = False
        
        # Set events to unblock any waiting threads
        self.sanity_check_audio_complete.set()
        self.sanity_check_generation_complete.set()
        
        if self.sanity_check_thread:
            self.sanity_check_thread.join(timeout=5.0)
        if self.playback_managing_thread:
            self.playback_managing_thread.join(timeout=2.0)
        if self.generation_thread:
            self.generation_thread.join(timeout=2.0)
        self.logger.info("Wrapper threads stopped")
    
    def reset_context(self):
        """
        Reset all state-related variables.
        
        This clears the current utterance content and stopped history.
        Useful for starting fresh or cleaning up after a session ends.
        """
        with self.utterance_lock:
            self.current_utterance.clear()
            self.stopped_utterances.clear()

    '''
    Utterance lifecycle methods
    '''

    def add_audio_chunk(self, utterance_id: int, chunk_index: int, audio_data: list, 
                        duration: float = None):
        """
        Add a new audio chunk from the dialogue system.
        Playback automatically starts when the first chunk (chunk_index=0) arrives.
        Only keeps the latest utterance - previous utterances are discarded.
        
        Args:
            utterance_id: Unique identifier for the utterance (msg_idx in your system)
            chunk_index: Position of this chunk within the utterance (starts from 0)
            audio_data: Raw audio data (list of integers)
            duration: Optional duration of the chunk in seconds (not used internally, kept for API compatibility)
        """

        # Reject chunks for stopped/timed-out utterances
        if utterance_id in self.stopped_utterances:
            # Silently ignore - this is expected for late-arriving chunks after stopping
            return
        
        curren_time = time.time()

        with self.utterance_lock:
            # Double-check after acquiring lock (in case it was stopped while we were creating the chunk)
            if utterance_id in self.stopped_utterances:
                return
            
            # If this is a new utterance, clear the old one and update ID
            if self.current_utterance.utterance_id != utterance_id:
                
                ## Stop the old utterance if it exists
                self.stop_current_utterance(will_lock=False)
                
                # Update utterance ID for the new utterance
                self.current_utterance.utterance_id = utterance_id
                self.logger.info(f"Utterance object changed to new id={utterance_id}")
            
            
            # Update last chunk received time
            self.current_utterance.last_chunk_received_time = curren_time
            
            # Automatically start playback when first chunk arrives, we log the timestamp here
            if chunk_index == 0 and self.current_utterance.start_time == Utterance.PLACE_HOLDER_TIMESTAMP:
                self.current_utterance.start_time = curren_time
                self.logger.info(f"Utterance {utterance_id} playback started at chunk 0")
            
            # Add audio samples to utterance
            audio_samples_before = self.current_utterance.get_total_samples()
            self.current_utterance.add_audio_samples(audio_data)
            audio_samples_after = self.current_utterance.get_total_samples()
            
            # # Avoid too frequent logging
            # self.logger.debug(f"Utterance {utterance_id} chunk {chunk_index}: added {audio_samples_after - audio_samples_before} samples (total: {audio_samples_after})")
    
    def stop_current_utterance(self, will_lock: bool):
        """
        Clear the current utterance content and prepare it for reuse.
        
        Since gesture generation is faster than realtime, it's safe to simply
        clear the current utterance and all its waypoints when stopped.
        The system will start fresh with the next utterance using the same object.
        
        This also prevents late-arriving chunks for this utterance from being
        processed by tracking stopped utterance IDs in a set.
        
        """
        if will_lock:
            self.utterance_lock.acquire()

        try:
            # Add to stopped set to reject any late-arriving chunks
            if self.current_utterance.utterance_id != Utterance.PLACE_HOLDER_ID:
                self.stopped_utterances.add(self.current_utterance.utterance_id)
                
                # Clear the utterance content but keep the object alive
                total_samples = self.current_utterance.get_total_samples()
                self.logger.debug(f"Stop utterance {self.current_utterance.utterance_id} (had {total_samples} samples)")
                
                # Clear all content for reuse
                self.current_utterance.clear()
                
        except Exception as e:
            self.logger.error(f'Failed to stop utterance with exception:{e}')

        finally:
            if will_lock:
                self.utterance_lock.release()



    '''
    Track utterance playback and manage gesture playback
    '''
    def _playback_managing_loop(self):
        """
        Thread 1: Playback Managing Thread
        
        This thread performs two main tasks:
        1. Track current time (wall clock and relative to audio start)
        2. Check if there's a waypoint to execute in the next 10ms interval
        
        Waypoint execution:
        - Gestures are generated as waypoints at 15 FPS (one every ~66.67ms)
        - This thread checks every 10ms for waypoints to execute
        - At most 1 waypoint per 10ms interval (since waypoints are 66.67ms apart)
        - Retrieves the waypoint if one should be executed in the upcoming interval
        
        Playback lifecycle:
        - Starts automatically when first chunk (index 0) arrives
        - Auto-cleanup after playback ends (default 2 seconds after last audio)
        """
        interval_duration = 0.01  # 10ms target interval
        
        while self.running:
            iteration_start_time = time.time()
            
            should_cleanup = False
            
            with self.utterance_lock:
                
                # Skip if no valid utterance (placeholder ID)
                if self.current_utterance.utterance_id == Utterance.PLACE_HOLDER_ID:
                    continue
                
                # Skip if playback hasn't started yet (no chunk 0 received)
                if self.current_utterance.start_time == Utterance.PLACE_HOLDER_TIMESTAMP:
                    continue
                
                # Calculate current playback position (time relative to utterance start)
                elapsed_time = iteration_start_time - self.current_utterance.start_time
                
                # Calculate total audio duration received so far
                total_samples = self.current_utterance.get_total_samples()
                total_audio_duration = total_samples / self.current_utterance.sample_rate
                

                # Auto-cleanup: if playback has passed all audio AND no new chunks for cleanup_timeout
                if elapsed_time > total_audio_duration:
                    time_since_last_chunk = iteration_start_time - self.current_utterance.last_chunk_received_time
                    if time_since_last_chunk > self.cleanup_timeout:
                        should_cleanup = True
                
                # Check for waypoint to execute in the next 10ms interval
                waypoint = self.current_utterance.get_waypoint_for_interval(
                    current_time=elapsed_time,
                    interval_duration=interval_duration
                )
                if waypoint is not None:
                    # Execute waypoint gesture
                    self.logger.debug(f"Utterance {self.current_utterance.utterance_id} executing waypoint {waypoint.waypoint_index} at t={elapsed_time:.3f}s (timestamp={waypoint.timestamp:.3f}s)")
                    # Call the waypoint callback if provided
                    if self.waypoint_callback is not None:
                        self.waypoint_callback(waypoint)
            
            if should_cleanup:
                self.stop_current_utterance(will_lock=True)
            
            # Sleep for the remaining time to maintain 10ms interval
            elapsed = time.time() - iteration_start_time
            sleep_time = max(0, interval_duration - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    

    '''
    Sanity check mode: Non-streaming batch generation
    '''
    
    def _sanity_check_loop(self):
        """
        Sanity check thread: Wait for all audio, then generate all waypoints at once.
        
        This mode tests the generation logic in a non-streaming setting, similar to
        the official offline generation. It helps isolate whether issues are due to
        the core generation or the streaming/windowing approach.
        
        Process:
        1. Monitor audio chunks and wait for completion (no new chunks for threshold)
        2. Generate all windows using FULL audio context (not limited to window end)
        3. Execute all waypoints through callback for saving/visualization
        """
        audio_complete_threshold = 0.5  # Wait 500ms after last chunk
        
        while self.running:
            time.sleep(0.1)  # 100ms tick
            
            # Wait for audio to start arriving
            with self.utterance_lock:
                if self.current_utterance.utterance_id == Utterance.PLACE_HOLDER_ID or self.current_utterance.start_time == Utterance.PLACE_HOLDER_TIMESTAMP:
                    continue
            
            # Monitor for audio completion
            while self.running:
                time.sleep(0.05)
                
                with self.utterance_lock:

                    if self.current_utterance.utterance_id == Utterance.PLACE_HOLDER_ID:
                        break
                    
                    # Check if we've stopped receiving chunks
                    if self.current_utterance.last_chunk_received_time == Utterance.PLACE_HOLDER_TIMESTAMP:
                        time_since_last_chunk = time.time() - self.current_utterance.last_chunk_received_time
                        if time_since_last_chunk >= audio_complete_threshold:
                            # Audio is complete!
                            utterance_id = self.current_utterance.utterance_id
                            total_samples = self.current_utterance.get_total_samples()
                            audio_duration = total_samples / self.current_utterance.sample_rate
                            
                            self.logger.info(
                                f"[SANITY CHECK] Utterance {utterance_id} audio complete: "
                                f"{total_samples} samples ({audio_duration:.2f}s)"
                            )
                            
                            # Signal that audio is complete
                            self.sanity_check_audio_complete.set()
                            break
            
            # Now generate all windows using FULL audio
            if self.sanity_check_audio_complete.is_set():
                self.logger.info("[SANITY CHECK] Starting full-audio generation...")
                self._sanity_check_generate_all()
                # Signal completion AFTER all waypoints have been executed
                self.logger.info("[SANITY CHECK] Signaling generation complete")
                self.sanity_check_generation_complete.set()
                break
    
    def _extract_audio_features(self, audio_bytes: bytes, sample_rate: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract mel spectrogram and HuBERT features from audio bytes.
        
        This implements the EXACT official feature extraction pipeline from
        trainers/ddpm_beat_trainer.py::test_custom_aud().
        
        Args:
            audio_bytes: Raw audio bytes (s16le encoded)
            sample_rate: Audio sample rate (e.g., 16000 Hz)
        
        Returns:
            Tuple of (mel_features, hubert_features):
                - mel_features: torch.Tensor [1, T, 128]
                - hubert_features: Optional[torch.Tensor] [1, T, 1024] (None if HuBERT disabled)
        """
        # Convert audio bytes to normalized float32
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        if audio_int16.size == 0:
            # Return empty tensors
            empty_mel = torch.zeros((1, 0, 128), device=self.device)
            return empty_mel, None
        
        aud_ori = audio_int16.astype(np.float32) / 32768.0
        
        # Resample to 18kHz for mel
        aud = librosa.resample(aud_ori, orig_sr=sample_rate, target_sr=18000)
        
        # Extract mel spectrogram (EXACT official code)
        mel = librosa.feature.melspectrogram(y=aud, sr=18000, hop_length=1200, n_mels=128)
        mel = mel[..., :-1]  # CRITICAL: Always remove last frame (official code line 1249)
        audio_emb = torch.from_numpy(np.swapaxes(mel, -1, -2))  # [T, 128]
        audio_emb = audio_emb.unsqueeze(0).to(self.device)  # [1, T, 128]
        
        # Extract HuBERT features if enabled
        hubert_feat = None
        if self.opt.expAddHubert or self.opt.addHubert:
            # Keep features on GPU to avoid unnecessary CPU-GPU transfers
            hubert_feat = get_hubert_from_16k_speech_long(
                self.hubert_model,
                self.wav2vec2_processor,
                torch.from_numpy(aud_ori).unsqueeze(0).to(self.device),
                device=self.device,
                return_on_cpu=False  # Keep on GPU for performance
            )
            # Interpolate to match mel length (already on GPU)
            hubert_feat = F.interpolate(
                hubert_feat.swapaxes(-1,-2).unsqueeze(0),
                size=audio_emb.shape[1],
                mode='linear',
                align_corners=True
            ).swapaxes(-1,-2)  # [1, T, 1024]
        
        return audio_emb, hubert_feat
    
    def _sanity_check_generate_all(self):
        """
        Generate all windows for the current utterance using full audio context.
        
        In non-streaming sanity check mode, we extract mel/HuBERT features ONCE for the full audio
        (like the official runner), then window them for each generation. This is more efficient
        and matches the official runner behavior exactly.
        """
        with self.utterance_lock:

            if self.current_utterance.utterance_id == Utterance.PLACE_HOLDER_ID:
                return
            
            utterance_id = self.current_utterance.utterance_id
            total_samples = self.current_utterance.get_total_samples()
            sample_rate = self.current_utterance.sample_rate
            gesture_fps = self.current_utterance.gesture_fps
            
            # Get FULL audio
            full_audio_bytes = self.current_utterance.get_audio_window(0, total_samples)
            
            self.logger.info(
                f"[SANITY CHECK] Generating all windows for utterance {utterance_id}: "
                f"total_samples={total_samples}, duration={total_samples/sample_rate:.2f}s"
            )
        
        # ===== STEP 1: Extract features ONCE for the full audio (like official runner) =====
        self.logger.info("[SANITY CHECK] Extracting mel/HuBERT features for full audio...")
        audio_emb, hubert_feat = self._extract_audio_features(full_audio_bytes, sample_rate)
        self.logger.info(f"[SANITY CHECK] Mel features extracted: shape={audio_emb.shape}")
        if hubert_feat is not None:
            self.logger.info(f"[SANITY CHECK] HuBERT features extracted: shape={hubert_feat.shape}")
        
        # ===== STEP 2: Generate all windows by windowing the precomputed features =====
        all_waypoints = []
        all_window_outputs = []  # Collect raw outputs for export
        overlap_context = None
        
        window_start_sample = 0
        window_end_sample = int((self.window_size / gesture_fps) * sample_rate)
        
        window_idx = 0
        while window_start_sample < total_samples:
            self.logger.info(
                f"[SANITY CHECK] Generating window {window_idx}: "
                f"samples [{window_start_sample}-{window_end_sample}]"
            )
            
            # Generate this window using precomputed features
            # Pass empty bytes since we're using precomputed features
            window = self._generate_gesture_window_from_audio(
                b'',  # Not used when precomputed features are provided
                window_start_sample,
                min(window_end_sample, total_samples),
                sample_rate,
                gesture_fps,
                overlap_context,
                utterance_id,
                precomputed_mel=audio_emb,
                precomputed_hubert=hubert_feat
            )
            
            if window is None:
                break
            
            # Collect execution waypoints for export
            all_waypoints.extend(window.execution_waypoints)
            
            # Collect gesture data from execution waypoints for export
            execution_data = np.array([wp.gesture_data for wp in window.execution_waypoints])  # Shape: (window_step, C)
            all_window_outputs.append(execution_data)
            
            # Update overlap context for next window using context waypoints
            if self.overlap_len > 0 and window.context_waypoints:
                # Extract gesture data from context waypoints (frames 30-33 of the window)
                overlap_context = [deepcopy(wp.gesture_data) for wp in window.context_waypoints]
                self.logger.debug(f"[SANITY CHECK] Updated overlap context with {len(overlap_context)} frames from window {window.window_index}")
            
            # Advance window
            window_start_sample += int((self.window_step / gesture_fps) * sample_rate)
            window_end_sample = window_start_sample + int((self.window_size / gesture_fps) * sample_rate)
            window_idx += 1
        
        self.logger.info(
            f"[SANITY CHECK] Generated {len(all_waypoints)} total waypoints "
            f"from {window_idx} windows"
        )
        
        # Concatenate all window outputs and store for export (matching reference pipeline)
        if all_window_outputs:
            out_motions = np.concatenate(all_window_outputs, axis=0)  # Shape: (T, C)
            out_motions = np.expand_dims(out_motions, axis=0)  # Shape: (1, T, C) to match reference pipeline
            self.last_generated_motion = torch.from_numpy(out_motions)
            self.logger.info(f"[SANITY CHECK] Stored generated motion for export: shape={out_motions.shape}")
        else:
            self.last_generated_motion = None
            self.logger.warning("[SANITY CHECK] No window outputs generated, cannot store motion for export")
        
        # Execute all waypoints through callback BEFORE signaling completion
        if self.waypoint_callback is not None:
            self.logger.info("[SANITY CHECK] Executing all waypoints through callback...")
            for waypoint in all_waypoints:
                self.waypoint_callback(waypoint)
                time.sleep(0.01)  
            self.logger.info("[SANITY CHECK] All waypoints executed")
        else:
            self.logger.warning("[SANITY CHECK] No waypoint callback provided, waypoints not executed")

    '''
    Gesture generation scheduling
    '''

    def __normalize_audio_via_wav_io(self, audio_bytes: bytes, sample_rate: int) -> bytes:
        """
        Normalize audio by saving to and reading from a temporary WAV file.
        This mimics what the audio preprocessing module does and may fix format issues.
        
        Args:
            audio_bytes: Raw audio bytes (s16le format)
            sample_rate: Sample rate (e.g., 24000)
        
        Returns:
            bytes: Normalized audio bytes after WAV I/O cycle
        """
        import tempfile
        import os
        
        try:
            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_path = temp_wav.name
                
                # Write audio to WAV file (this validates and normalizes the data)
                with wave.open(temp_path, 'wb') as wav_file:
                    wav_file.setnchannels(self.EXPECTED_CHANNELS)
                    wav_file.setsampwidth(2)  # 16-bit = 2 bytes
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_bytes)
                
                # Read audio back from WAV file
                with wave.open(temp_path, 'rb') as wav_file:
                    normalized_bytes = wav_file.readframes(wav_file.getnframes())
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            return normalized_bytes
            
        except Exception as e:
            self.logger.error(f"Failed to normalize audio via WAV I/O: {e}")
            return audio_bytes  # Return original if normalization fails

    def __normalize_audio_direct(self, audio_bytes: bytes) -> bytes:
        """
        Normalize audio without file I/O by ensuring proper format.
        """
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Ensure proper byte order and contiguity
            audio_array = np.ascontiguousarray(audio_array, dtype=np.int16)
            
            # Convert back to bytes with explicit byte order
            normalized_bytes = audio_array.tobytes()
            
            return normalized_bytes
            

            # """Ensure audio bytes are in contiguous memory with proper alignment."""
            # audio_array = np.frombuffer(audio_bytes, dtype=np.int16).copy()
            # return audio_array.tobytes()

        except Exception as e:
            self.logger.error(f"Failed to normalize audio directly: {e}")
            return audio_bytes


    def _generation_loop(self):
        """
        Thread 2: Monitor available audio and trigger gesture generation windows.
        
        Generation strategy:
        - Window indices are initialized automatically when utterance is created (starts from sample 0)
        - Wait until enough audio samples are available to fill next_window_end_sample
        - Snapshot truncated audio [0, window_end], release lock, and generate gestures
        - After generation, acquire lock and write waypoints back (if utterance still exists)
        - Update window indices using utterance.update_window_indices() for next generation
        
        Audio truncation for streaming:
        - Snapshots audio from [0, window_end] (as if no future audio exists)
        - Generation method extracts mel/HuBERT features from this truncated audio
        - Features are then windowed to get the current window portion
        - This ensures streaming compatibility while using official generation code
        
        Desired behaviors:
        - Starts generation from t=0 (first window begins at sample 0)
        - If utterance tail is insufficient for a full window, no generation is triggered
        - Generation naturally stays ahead of playback since it's faster than realtime
        """
        while self.running:
            time.sleep(0.05)  # 50ms tick
            
            # Step 1: Check if we have enough samples for the next window and snapshot everything needed
            should_generate = False
            utterance_id = None
            audio_snapshot_full = None  # CHANGED: Need full audio up to window end, not just window
            window_start_sample = None
            window_end_sample = None
            sample_rate = None
            gesture_fps = None
            overlap_context = None  # Snapshot of overlap waypoints for smooth transitions
            
            with self.utterance_lock:
                
                # Skip if no valid utterance (placeholder ID)
                if self.current_utterance.utterance_id == Utterance.PLACE_HOLDER_ID:
                    continue
                
                # Check if we have enough samples for the next window
                available_samples = self.current_utterance.get_total_samples()
                
                if available_samples < self.current_utterance.next_window_end_sample:
                    # Not enough samples yet, wait for more
                    continue
                
                # We have enough samples, prepare to generate
                should_generate = True
                utterance_id = self.current_utterance.utterance_id
                window_start_sample = self.current_utterance.next_window_start_sample
                window_end_sample = self.current_utterance.next_window_end_sample
                sample_rate = self.current_utterance.sample_rate
                gesture_fps = self.current_utterance.gesture_fps
                
                # Determine audio snapshot range based on USE_CONSTRAINED_FEATURES flag
                if self.USE_CONSTRAINED_FEATURES:
                    # Constrained mode: Use only AUDIO_DUR_FOR_FEATURES seconds of audio
                    constrained_samples = int(self.AUDIO_DUR_FOR_FEATURES * sample_rate)
                    # Take audio from [window_end - constrained_samples, window_end]
                    # But ensure we don't go negative
                    audio_start_sample = max(0, window_end_sample - constrained_samples)
                    audio_snapshot_full = self.current_utterance.get_audio_window(audio_start_sample, window_end_sample)
                    
                    window_duration = (window_end_sample - window_start_sample) / sample_rate
                    constrained_duration = (window_end_sample - audio_start_sample) / sample_rate
                    self.logger.debug(
                        f"Utterance {utterance_id} generation triggered (CONSTRAINED): "
                        f"window [{window_start_sample}-{window_end_sample}] ({window_duration:.3f}s), "
                        f"audio_context [{audio_start_sample}-{window_end_sample}] ({constrained_duration:.3f}s), "
                        f"available_samples={available_samples}"
                    )
                else:
                    # Default mode: Snapshot ALL audio from start to window end (not just the window)
                    # This is needed because mel/HuBERT features must be computed over full context
                    # Following official code pattern: compute features for entire audio, then window them
                    audio_start_sample = 0
                    audio_snapshot_full = self.current_utterance.get_audio_window(0, window_end_sample)
                    
                    window_duration = (window_end_sample - window_start_sample) / sample_rate
                    self.logger.debug(
                        f"Utterance {utterance_id} generation triggered: "
                        f"window [{window_start_sample}-{window_end_sample}] ({window_duration:.3f}s), "
                        f"available_samples={available_samples}, full_context_samples={window_end_sample}"
                    )
                
                # Snapshot overlap context if needed for smooth transitions
                # Frames are gesture poses indexed at gesture_fps (e.g., 15 FPS for BEAT)
                # Frame index = (sample_index / sample_rate) * gesture_fps
                # For example: at 16000 Hz and 15 FPS, 1 frame spans ~1067 samples
                window_start_frame = int(window_start_sample / sample_rate * gesture_fps)
                
                if self.overlap_len > 0 and window_start_frame > 0:
                    # We need overlap context from the previous window's context waypoints
                    if self.current_utterance.windows:
                        prev_window = self.current_utterance.windows[-1]
                        # Use the context waypoints from previous window (frames 30-33)
                        if prev_window.context_waypoints:
                            overlap_context = [deepcopy(wp.gesture_data) for wp in prev_window.context_waypoints]
                            self.logger.debug(f"Utterance {utterance_id} using overlap context from previous window {prev_window.window_index}")
            
            # Step 2: Generate gestures without holding the lock
            if should_generate and len(audio_snapshot_full) > 0:
                # Override with test audio if debugging flag is enabled
                if self.USE_TEST_DATA_FOR_WINDOW and self.test_audio_window is not None:
                    self.logger.debug(f"[TEST MODE] Using pre-loaded test audio window instead of real audio")
                    audio_snapshot_full = self.test_audio_window
                    # Reset context start to 0 since test window always starts from beginning
                    audio_start_sample = 0
                
                # Generate gestures for this window
                gen_start_time = time.time()
                window = self._generate_gesture_window_from_audio(
                    audio_snapshot_full,  # Pass audio context (or test data if flag enabled)
                    window_start_sample,
                    window_end_sample,
                    sample_rate,
                    gesture_fps,
                    overlap_context,
                    utterance_id,
                    audio_context_start_sample=audio_start_sample  # Pass context start for proper windowing
                )
                gen_duration = time.time() - gen_start_time
                if window:
                    self.logger.debug(f"Utterance {utterance_id} generation completed: window {window.window_index} with {len(window.execution_waypoints)} execution waypoints in {gen_duration:.3f}s")
                
                # Step 3: Write window back and update generation state
                with self.utterance_lock:
                    utterance = self.current_utterance
                    
                    # Check if this utterance is still current (not stopped)
                    if self.current_utterance.utterance_id != utterance_id:
                        # Utterance was stopped, discard window
                        self.logger.debug(f"Utterance {utterance_id} was stopped during generation, discarding window")
                        continue
                    
                    # Add window to utterance (this also adds execution waypoints to flat list)
                    if window:
                        self.current_utterance.add_window(window)
                        self.logger.debug(f"Utterance {utterance_id} window {window.window_index} added: total windows={len(self.current_utterance.windows)}, total execution waypoints={len(self.current_utterance.execution_waypoints)}")
                    
                    # Update window indices for next generation
                    prev_window_end = self.current_utterance.next_window_end_sample
                    self.current_utterance.update_window_indices()
                    self.logger.debug(f"Utterance {utterance_id} window updated: next_window=[{self.current_utterance.next_window_start_sample}-{self.current_utterance.next_window_end_sample}] (step={self.current_utterance.next_window_start_sample - (prev_window_end - (self.current_utterance.next_window_end_sample - self.current_utterance.next_window_start_sample))} samples)")

    def _generate_gesture_window_from_audio(
        self, 
        audio_bytes_truncated: bytes,
        window_start_sample: int,
        window_end_sample: int,
        sample_rate: int,
        gesture_fps: int,
        overlap_context: Optional[List[np.ndarray]],
        utterance_id: int,
        precomputed_mel: Optional[torch.Tensor] = None,
        precomputed_hubert: Optional[torch.Tensor] = None,
        audio_context_start_sample: int = 0
    ) -> WaypointWindow:
        """
        Generate gestures for a single window using official DiffSHEG pipeline.
        
        This method implements the EXACT official generation code from 
        trainers/ddpm_beat_trainer.py::test_custom_aud(), adapted for single-window generation.
        
        KEY STRATEGY FOR STREAMING:
        - audio_bytes_truncated contains audio from [audio_context_start_sample, window_end_sample]
        - We compute mel/HuBERT features for this audio context
        - We then window these features to extract the current window portion
        - This ensures temporal consistency while enabling streaming
        
        CONSTRAINED FEATURES MODE:
        - When USE_CONSTRAINED_FEATURES=True, audio_context_start_sample may be > 0
        - Features are extracted from constrained audio context [audio_context_start_sample, window_end_sample]
        - Window indexing accounts for the offset via audio_context_start_sample parameter
        
        NON-STREAMING MODE (precomputed_mel/hubert provided):
        - Features are extracted once for the full audio
        - Each window just indexes into the precomputed features
        - This matches official runner behavior (one feature extraction, multiple windows)
        
        Official DiffSHEG pipeline:
        1. Convert audio bytes to float32 (normalized to [-1, 1])
        2. Resample to 18kHz for mel spectrogram
        3. Extract mel: librosa.feature.melspectrogram(...) then mel[..., :-1]
        4. Extract HuBERT if enabled: get_hubert_from_16k_speech_long(...) then interpolate
        5. Window the features to get current window (window_size frames)
        6. Generate with trainer.generate_batch() using inpainting for overlap
        
        Args:
            audio_bytes_truncated: Raw audio bytes (s16le encoded)
                                   In streaming mode: from [audio_context_start_sample, window_end_sample]
                                   In non-streaming mode: can be ignored if precomputed features are provided
            window_start_sample: Starting sample index of this window in the full utterance
            window_end_sample: Ending sample index of this window (where audio is truncated)
            sample_rate: Audio sample rate (e.g., 16000 Hz)
            gesture_fps: Gesture frame rate (e.g., 15 FPS for BEAT)
            overlap_context: Optional list of gesture data arrays from previous window's last frames
                           Used for smooth transitions via inpainting. Length should be overlap_len.
            utterance_id: ID of the utterance (for debugging/logging)
            precomputed_mel: Optional precomputed mel features [1, T, 128] for non-streaming mode
            precomputed_hubert: Optional precomputed HuBERT features [1, T, 1024] for non-streaming mode
            audio_context_start_sample: Starting sample index of the audio context (default 0)
                                       Used to correctly window features when using constrained context
        
        Returns:
            WaypointWindow containing execution waypoints and context waypoints
        """
        if len(audio_bytes_truncated) == 0 and precomputed_mel is None:
            return None

        # ===== SAVE AUDIO WINDOW FOR DEBUGGING (if enabled) =====
        window_idx = int(round(window_start_sample / sample_rate * gesture_fps)) // self.window_step
        self._save_audio_window(
            audio_bytes=audio_bytes_truncated,
            sample_rate=sample_rate,
            utterance_id=utterance_id,
            window_index=window_idx,
            window_start_sample=window_start_sample,
            window_end_sample=window_end_sample
        )
        
        # ===== STEP 1-4: Extract or use precomputed features =====
        self.logger.debug('Extracting audio features for the window.')
        
        # Disable GC during feature extraction to prevent interference with CUDA operations
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
        
        try:
            if precomputed_mel is not None:
                # Non-streaming mode: Use precomputed features
                audio_emb = precomputed_mel  # [1, T, 128]
                add_cond = {}
                if precomputed_hubert is not None:
                    add_cond["pretrain_aud_feat"] = precomputed_hubert  # [1, T, 1024]
            else:
                # Streaming mode: Extract features from truncated audio using helper
                audio_emb, hubert_feat = self._extract_audio_features(audio_bytes_truncated, sample_rate)
                if audio_emb.shape[1] == 0:
                    return []
                
                add_cond = {}
                if hubert_feat is not None:
                    add_cond["pretrain_aud_feat"] = hubert_feat
        finally:
            # Re-enable GC if it was enabled before
            if gc_was_enabled:
                gc.enable()
        
        # ===== STEP 5: Window the features to get current window portion =====
        window_start_frame = int(round(window_start_sample / sample_rate * gesture_fps))
        audio_context_start_frame = int(round(audio_context_start_sample / sample_rate * gesture_fps))
        
        # Extract window from features
        # Features span [audio_context_start_sample, window_end_sample]
        # So we need to offset the window indices by audio_context_start_frame
        mel_window_start = window_start_frame - audio_context_start_frame
        mel_window_end = mel_window_start + self.window_size
        
        audio_window = audio_emb[:, mel_window_start:mel_window_end, :]
        
        # Pad or trim to exact window_size
        if audio_window.shape[1] < self.window_size:
            pad_frames = self.window_size - audio_window.shape[1]
            audio_window = F.pad(audio_window, (0, 0, 0, pad_frames), value=0)
        elif audio_window.shape[1] > self.window_size:
            audio_window = audio_window[:, :self.window_size, :]
        
        audio_window = audio_window.contiguous()  # [1, window_size, 128]
        
        # Window HuBERT features similarly if present
        if "pretrain_aud_feat" in add_cond:
            hubert_window = add_cond["pretrain_aud_feat"][:, mel_window_start:mel_window_end, :]
            if hubert_window.shape[1] < self.window_size:
                pad_frames = self.window_size - hubert_window.shape[1]
                hubert_window = F.pad(hubert_window, (0, 0, 0, pad_frames), value=0)
            elif hubert_window.shape[1] > self.window_size:
                hubert_window = hubert_window[:, :self.window_size, :]
            add_cond["pretrain_aud_feat"] = hubert_window.contiguous()
        
        # ===== STEP 6: Prepare motion tensor and person ID =====
        B, T, _ = audio_window.shape
        C = self.opt.net_dim_pose
        motions = torch.zeros((B, T, C), device=self.device)
        
        # Person ID (use ID 2, index 1 - same as official test_custom_aud)
        p_id = torch.ones((1, 1), device=self.device) * 1  # pid 2 - 1 = 1
        p_id = self.model.one_hot(p_id, self.opt.speaker_dim).to(self.device)
        
        # ===== STEP 7: Prepare inpainting dict for overlap (EXACT official code) =====
        inpaint_dict = {}
        if self.overlap_len > 0:
            inpaint_dict['gt'] = torch.zeros_like(motions)
            inpaint_dict['outpainting_mask'] = torch.zeros_like(
                motions, dtype=torch.bool, device=self.device
            )
            
            # Use overlap context from previous window (if available)
            # Note: For the very first window (window_start_frame == 0), overlap_context will be None,
            # so no inpainting is applied. This matches the official runner behavior (fix_very_first=False).
            if overlap_context is not None and len(overlap_context) == self.overlap_len:
                inpaint_dict['outpainting_mask'][:, :self.overlap_len, :] = True
                prev_frames_np = np.stack(overlap_context).astype(np.float32)
                prev_frames = torch.from_numpy(prev_frames_np).unsqueeze(0).to(self.device)
                inpaint_dict['gt'][:, :self.overlap_len, :] = prev_frames
        
        # ===== STEP 8: Generate using official trainer method =====
        self.logger.debug('Starting generation for the window.')
        
        # Disable GC during CUDA operations to prevent memory allocation stalls
        # This is critical when ENAGLE_CLEARING_META2=True as GC can interfere
        # with CuDNN kernel selection and GPU memory allocation
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
        
        try:
            with torch.no_grad():

                ## Ensure previous operations are complete without blocking
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)  # Clear any pending ops
                

                outputs = self.model.generate_batch(
                    audio_window, p_id, C, add_cond, inpaint_dict
                )
        finally:
            # Re-enable GC if it was enabled before
            if gc_was_enabled:
                gc.enable()
        

        self.logger.debug('Finalizing generated outputs for the window.')

        outputs_np = outputs.cpu().numpy()[0]  # [window_size, C] - FULL window output
        
        # ===== STEP 9: Create individual waypoints for each frame =====
        execution_waypoints = []
        context_waypoints = []
        
        # Create execution waypoints (first window_step frames)
        for i in range(self.window_step):
            frame_index = window_start_frame + i
            timestamp = frame_index / gesture_fps
            
            waypoint = GestureWaypoint(
                waypoint_index=frame_index,
                timestamp=timestamp,
                gesture_data=outputs_np[i],  # [C,] - single frame
                is_for_execution=True
            )
            execution_waypoints.append(waypoint)
        
        # Create context waypoints (last overlap_len frames)
        if self.overlap_len > 0:
            for i in range(self.overlap_len):
                frame_index = window_start_frame + self.window_step + i
                timestamp = frame_index / gesture_fps
                
                waypoint = GestureWaypoint(
                    waypoint_index=frame_index,
                    timestamp=timestamp,
                    gesture_data=outputs_np[self.window_step + i],  # [C,] - single frame
                    is_for_execution=False  # Context only, not for execution
                )
                context_waypoints.append(waypoint)
        
        # Create window containing all waypoints
        window_idx = window_start_frame // self.window_step
        window = WaypointWindow(
            window_index=window_idx,
            execution_waypoints=execution_waypoints,
            context_waypoints=context_waypoints
        )
        
        self.logger.debug(
            f"Utterance {utterance_id} window {window_idx}: "
            f"generated {len(execution_waypoints)} execution waypoints + {len(context_waypoints)} context waypoints"
        )
        return window

    
