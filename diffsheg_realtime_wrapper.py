"""
DiffSHEG Realtime Wrapper for GPT-4o Dialogue System Integration

This wrapper class manages the integration of DiffSHEG gesture generation with
a real-time dialogue system that receives audio chunks from GPT-4o-realtime.


- Real-time gesture generation as audio chunks arrive
- Uses incremental windowing with overlap context
- Audio is truncated to [0, window_end] for each window generation
   

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
import traceback
import tempfile
import os
from copy import deepcopy
from PlaybackState import PlaybackState

# Add parent directory to path to import logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger.logger import setup_logger
import wave
from copy import deepcopy

try:
    from trainers.ddpm_beat_trainer import get_hubert_from_16k_speech_long
except ImportError:
    get_hubert_from_16k_speech_long = None


DO_AUD_NORMALIZATION = True




def normalize_audio_via_wav_io(audio_bytes: bytes, sample_rate: int) -> bytes:
    """
    Normalize audio by saving to and reading from a temporary WAV file.
    This mimics what the audio preprocessing module does and may fix format issues.
    
    Args:
        audio_bytes: Raw audio bytes (s16le format)
        sample_rate: Sample rate (e.g., 24000)
    
    Returns:
        bytes: Normalized audio bytes after WAV I/O cycle
    """
    
    try:
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_path = temp_wav.name
            
            # Write audio to WAV file (this validates and normalizes the data)
            with wave.open(temp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
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
        print(f"Failed to normalize audio via WAV I/O: {e}")
        return audio_bytes  # Return original if normalization fails

def normalize_audio_direct(audio_bytes: bytes) -> bytes:
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
        print(f"Failed to normalize audio directly: {e}")
        return audio_bytes




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


# BEAT skeleton joint order as defined in DiffSHEG data_tools.py
# This defines which 47 joints are used from the full skeleton, in order
BEAT_GESTURE_JOINT_ORDER = [
    'Spine',          # indices 0-2
    'Neck',           # indices 3-5
    'Neck1',          # indices 6-8
    
    # Right arm
    'RightShoulder',  # indices 9-11
    'RightArm',       # indices 12-14
    'RightForeArm',   # indices 15-17
    'RightHand',      # indices 18-20
    
    # Right hand fingers - Middle
    'RightHandMiddle1',  # indices 21-23
    'RightHandMiddle2',  # indices 24-26
    'RightHandMiddle3',  # indices 27-29
    
    # Right hand - Ring
    'RightHandRing',     # indices 30-32
    'RightHandRing1',    # indices 33-35
    'RightHandRing2',    # indices 36-38
    'RightHandRing3',    # indices 39-41
    
    # Right hand - Pinky
    'RightHandPinky',    # indices 42-44
    'RightHandPinky1',   # indices 45-47
    'RightHandPinky2',   # indices 48-50
    'RightHandPinky3',   # indices 51-53
    
    # Right hand - Index
    'RightHandIndex',    # indices 54-56
    'RightHandIndex1',   # indices 57-59
    'RightHandIndex2',   # indices 60-62
    'RightHandIndex3',   # indices 63-65
    
    # Right hand - Thumb
    'RightHandThumb1',   # indices 66-68
    'RightHandThumb2',   # indices 69-71
    'RightHandThumb3',   # indices 72-74
    
    # Left arm
    'LeftShoulder',      # indices 75-77
    'LeftArm',           # indices 78-80
    'LeftForeArm',       # indices 81-83
    'LeftHand',          # indices 84-86
    
    # Left hand fingers - Middle
    'LeftHandMiddle1',   # indices 87-89
    'LeftHandMiddle2',   # indices 90-92
    'LeftHandMiddle3',   # indices 93-95
    
    # Left hand - Ring
    'LeftHandRing',      # indices 96-98
    'LeftHandRing1',     # indices 99-101
    'LeftHandRing2',     # indices 102-104
    'LeftHandRing3',     # indices 105-107
    
    # Left hand - Pinky
    'LeftHandPinky',     # indices 108-110
    'LeftHandPinky1',    # indices 111-113
    'LeftHandPinky2',    # indices 114-116
    'LeftHandPinky3',    # indices 117-119
    
    # Left hand - Index
    'LeftHandIndex',     # indices 120-122
    'LeftHandIndex1',    # indices 123-125
    'LeftHandIndex2',    # indices 126-128
    'LeftHandIndex3',    # indices 129-131
    
    # Left hand - Thumb
    'LeftHandThumb1',    # indices 132-134
    'LeftHandThumb2',    # indices 135-137
    'LeftHandThumb3',    # indices 138-140
]


def build_joint_mask_indices(joint_names: List[str]) -> List[int]:
    """
    Convert a list of joint names to their corresponding dimension indices in the 141-dimensional gesture data.
    
    Args:
        joint_names: List of joint names from BEAT_GESTURE_JOINT_ORDER, e.g., ['Spine', 'Neck', 'RightArm']
    
    Returns:
        List of dimension indices (0-140) corresponding to the requested joints.
        Each joint contributes 3 dimensions (X, Y, Z rotations), e.g., 'Spine' -> [0, 1, 2]
    
    Raises:
        ValueError: If any joint name is not found in the BEAT skeleton
    """
    if not joint_names:
        return list(range(141))  # Use all dimensions if no mask specified
    
    mask_indices = []
    for joint_name in joint_names:
        if joint_name not in BEAT_GESTURE_JOINT_ORDER:
            raise ValueError(
                f"Joint '{joint_name}' not found in BEAT skeleton. "
                f"Available joints: {', '.join(BEAT_GESTURE_JOINT_ORDER)}"
            )
        
        joint_idx = BEAT_GESTURE_JOINT_ORDER.index(joint_name)
        # Each joint has 3 dimensions (XYZ rotations)
        dimension_start = joint_idx * 3
        mask_indices.extend([dimension_start, dimension_start + 1, dimension_start + 2])
    
    return sorted(mask_indices)


def build_neutral_position_array(
    joint_mask_names: List[str],
    custom_neutral_positions: Dict[str, List[float]] = None,
    net_dim_pose: int = 192,
    split_pos: int = 141
) -> np.ndarray:
    """
    Build a neutral position array for gestures covering both body and face.
    
    For BEAT dataset:
    - net_dim_pose = 192 (default): 141 gesture + 51 expression
    - split_pos = 141: divides gesture (0:141) from expression (141:192)
    
    Args:
        joint_mask_names: List of joint names in the mask (from config). Only applies to gesture portion.
        custom_neutral_positions: Optional dict mapping joint names to [x, y, z] neutral angles in degrees.
                                  If a joint is in the mask but not in custom_neutral_positions, defaults to [0, 0, 0].
        net_dim_pose: Total output dimension from model (default 192 for BEAT: 141 gesture + 51 expression)
        split_pos: Split position between gesture and expression (default 141 for BEAT)
    
    Returns:
        np.ndarray of shape (net_dim_pose,) with neutral positions.
        - Gesture portion (0:split_pos): Masked joints get their custom/default values, unmasked get 0
        - Expression portion (split_pos:net_dim_pose): All dimensions set to 0
    
    Raises:
        ValueError: If a joint in custom_neutral_positions is not in the joint mask
    """
    custom_neutral_positions = custom_neutral_positions or {}
    
    # Initialize all dimensions to 0 (covers both gesture and expression)
    neutral_array = np.zeros(net_dim_pose, dtype=np.float32)
    
    # Validate that all custom positions are for joints in the mask
    for joint_name in custom_neutral_positions.keys():
        if joint_name not in joint_mask_names:
            print(
                f"Joint '{joint_name}' in custom_neutral_positions is not in joint_mask. Will ignore it."
            )
    
    # Set neutral positions for masked joints in the gesture portion only
    for joint_name in joint_mask_names:
        joint_idx = BEAT_GESTURE_JOINT_ORDER.index(joint_name)
        dimension_start = joint_idx * 3
        
        # Ensure we're within gesture bounds
        if dimension_start + 3 > split_pos:
            raise ValueError(
                f"Joint '{joint_name}' extends beyond gesture split position {split_pos}"
            )
        
        # Use custom position if provided, otherwise default to [0, 0, 0]
        if joint_name in custom_neutral_positions:
            neutral_pos = custom_neutral_positions[joint_name]
            if not isinstance(neutral_pos, (list, tuple)) or len(neutral_pos) != 3:
                raise ValueError(
                    f"Invalid neutral position for '{joint_name}': expected [x, y, z] list/tuple, "
                    f"got {neutral_pos}"
                )
            neutral_array[dimension_start:dimension_start + 3] = neutral_pos
        else:
            # Default to [0, 0, 0] for masked joints without custom position
            neutral_array[dimension_start:dimension_start + 3] = [0, 0, 0]
    
    # Expression portion (split_pos:net_dim_pose) remains 0 (already initialized)
    
    return neutral_array


def apply_joint_mask_to_waypoint(
    waypoint: GestureWaypoint,
    mask_indices: List[int],
    split_pos: int = 141,
    custom_neural_positions_for_masked = None
) -> GestureWaypoint:
    """
    Apply a joint mask to a waypoint, zeroing out all non-masked joints.
    
    Works with full 192-dimensional gesture data (141 gesture + 51 expression):
    - Only masks the gesture portion (dims 0:split_pos)
    - Preserves the expression portion (dims split_pos:192) unchanged
    
    Args:
        waypoint: The original gesture waypoint with net_dim_pose dimensions (typically 192)
        mask_indices: List of dimension indices to preserve in gesture portion (from build_joint_mask_indices)
        split_pos: Position dividing gesture (0:split_pos) from expression (split_pos:end). Default 141.
        custom_neural_positions_for_masked: Debug entry. If supplied, all masked joints will use their custom neutral position value, discarding the waypoint information
    Returns:
        A new GestureWaypoint with masked gesture data (non-masked gesture joints set to 0,
        expression portion unchanged)
    """
    if not mask_indices or len(mask_indices) == split_pos:
        # No masking needed
        return waypoint
    
    gesture_dim = waypoint.gesture_data.shape[0]
    
    # Start with copy of original waypoint data
    masked_gesture = waypoint.gesture_data.copy()
    
    # Extract gesture and expression portions
    gesture_portion = masked_gesture[:split_pos]
    expression_portion = masked_gesture[split_pos:] if gesture_dim > split_pos else np.array([])
    
    # Zero out non-masked dimensions -- note that this is in the normalized space
    masked_gesture_portion = np.zeros(split_pos, dtype=np.float32)


    for mask_idx in mask_indices:
        if mask_idx < split_pos:
            if custom_neural_positions_for_masked is not None:## Force custom neutral position for all masked gesture dimensions
                masked_gesture_portion[mask_idx] = custom_neural_positions_for_masked[mask_idx]
            else:## Copy masked dimensions from original joint angles
                masked_gesture_portion[mask_idx] = gesture_portion[mask_idx]



    # Reconstruct full gesture with masked gesture + unchanged expression
    masked_gesture[:split_pos] = masked_gesture_portion
    # Expression portion remains unchanged from original
    
    # Create and return new waypoint with masked data
    return GestureWaypoint(
        waypoint_index=waypoint.waypoint_index,
        timestamp=waypoint.timestamp,
        gesture_data=masked_gesture,
        is_for_execution=waypoint.is_for_execution
    )


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
        window_step: int,
        generation_delayed_start_sec: Optional[float] = 0
    ):
        self.utterance_id = utterance_id
        self.start_time: Optional[float] = Utterance.PLACE_HOLDER_TIMESTAMP

        # Audio storage: concatenated samples instead of chunks
        self.sample_rate = sample_rate
        self.audio_samples: bytes = b''  # Concatenated raw audio samples (s16le encoded)
        self.bytes_per_sample = 2  # s16le encoding uses 2 bytes per sample

        # Generation state: tracks which audio window to generate next (in terms of sample indices)
        self.gesture_fps = gesture_fps
        self.window_size = window_size  # Number of frames per window
        self.window_step = window_step  # Number of non-overlapping frames per window

        # Initialize window indices - start from sample 0 or delayed start
        window_duration_samples = int((window_size / gesture_fps) * sample_rate)

        self.generation_delayed_start_sec = generation_delayed_start_sec
        generation_delayed_start_sample_cnt = int(self.generation_delayed_start_sec * sample_rate)
        self.next_window_start_sample: int = generation_delayed_start_sample_cnt

        self.next_window_end_sample: int = self.next_window_start_sample + window_duration_samples

        # Gesture data structures
        self.windows: List[WaypointWindow] = []  # All generated windows
        self.execution_waypoints: List[GestureWaypoint] = []  # Flat list of waypoints for execution (only is_for_execution=True)
        self.last_executed_waypoint_index: int = -1  # Cursor for playback
        self.waypoints_lock = threading.Lock()
        
        # Utterance duration tracking
        self.total_duration: Optional[float] = None  # Total duration of the utterance in seconds
    
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
    
    def initialize_new_utterance(self, utterance_id: int, start_time: float, 
                                 prefill_neutral_window: Optional[WaypointWindow] = None):
        """
        Initialize a new utterance with ID and start time.
        
        If generation_delayed_start_sec > 0, pre-fills the windows with a neutral position window
        to provide playable content during the delayed start period.
        
        Args:
            utterance_id: Unique identifier for this utterance
            start_time: Wall clock time when playback starts (seconds)
            prefill_neutral_window: Optional pre-constructed neutral window to use for delayed start.
                                   This method will set proper timestamps based on the gesture FPS.
        """
        with self.waypoints_lock:
            self.utterance_id = utterance_id
            self.start_time = start_time
            
            # Pre-fill with neutral window if delayed start is configured
            if self.generation_delayed_start_sec > 0 and prefill_neutral_window is not None:
                # Create a deep copy to avoid modifying the template
                neutral_window_copy = deepcopy(prefill_neutral_window)
                
                # Add the prefilled window to windows and execution waypoints
                self.windows.append(neutral_window_copy)
                self.execution_waypoints.extend(neutral_window_copy.execution_waypoints)
    
    def clear(self):
        """
        Clear all content of the utterance for reuse.
        This resets all data structures while keeping the object alive.
        """

    
        with self.waypoints_lock:
            
            self.utterance_id = Utterance.PLACE_HOLDER_ID  # Reset to placeholder ID

            # Reset window indices to start from sample 0 or delayed start
            window_duration_samples = int((self.window_size / self.gesture_fps) * self.sample_rate)
            generation_delayed_start_sample_cnt = int(self.generation_delayed_start_sec * self.sample_rate)

            self.next_window_start_sample = generation_delayed_start_sample_cnt
            self.next_window_end_sample = self.next_window_start_sample + window_duration_samples

                
            # Clear timing information
            self.start_time = Utterance.PLACE_HOLDER_TIMESTAMP
            self.total_duration = None  # Reset utterance duration

            # Clear audio data
            self.audio_samples = b''
            
            # Clear gesture data structures
            self.windows.clear()
            self.execution_waypoints.clear()
            self.last_executed_waypoint_index = -1
                
    def is_active(self) -> bool:
        """
        Check whether this utterance is currently active.
        
        An utterance is considered active if:
        - It has a valid utterance ID (not placeholder)
        - It has a valid start time (not placeholder)
        
        Returns:
            True if the utterance is active, False otherwise.
        """
        return (self.utterance_id != Utterance.PLACE_HOLDER_ID and 
                self.start_time != Utterance.PLACE_HOLDER_TIMESTAMP)
    
    def get_waypoint_to_execute_for_interval(self, current_time: float, interval_duration: float = 0.01) -> Optional[GestureWaypoint]:
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
    
    USE_CONSTRAINED_FEATURES mode:
    - When False (default), streaming mode computes features from [0, window_end]
    - When True, constrains audio context to AUDIO_DUR_FOR_FEATURES seconds
    - This ensures stable performance regardless of utterance length
    """
    
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
            waypoint_callback: Optional callback function to execute waypoints.
                             Should accept a GestureWaypoint object as parameter.
                             If None, waypoints are generated but not executed.
        """
        # Initialize logger
        self.logger = setup_logger(
            logger_name='diffsheg_realtime_wrapper',
            file_log_level="DEBUG",
            terminal_log_level="INFO"
        )


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

        # Speaker ID configuration (for BEAT: 0-29)
        self.speaker_id = gesture_config.get('speaker_id', 0)
        if not isinstance(self.speaker_id, int) or not (self.speaker_id in [2,4,6,8]):
            raise ValueError(f"Invalid speaker_id {self.speaker_id}. Must be integer in range 0-29 for BEAT dataset.")
            
            
        # Store waypoint callback
        self.waypoint_callback = waypoint_callback
        
        # Get model output dimensions from opt
        self.net_dim_pose = getattr(opt, 'net_dim_pose', 192)  # Default 192 for BEAT: 141 gesture + 51 expression
        self.split_pos = getattr(opt, 'split_pos', 141)  # Default 141: gesture/expression split position
        
        # Load and build joint mask for gesture output
        joint_mask_names = gesture_config.get('joint_mask', [])
        try:
            self.joint_mask_indices = build_joint_mask_indices(joint_mask_names)
        except ValueError as e:
            raise ValueError(f"Invalid joint_mask configuration: {e}")
        
        # Load and build neutral position array
        custom_neutral_positions = gesture_config.get('custom_neutral_positions', {})
        try:
            self.neutral_position = build_neutral_position_array(
                joint_mask_names, 
                custom_neutral_positions,
                net_dim_pose=self.net_dim_pose,
                split_pos=self.split_pos
            )
        except ValueError as e:
            raise ValueError(f"Invalid custom_neutral_positions configuration: {e}")
        
        if self.joint_mask_indices and len(self.joint_mask_indices) < self.split_pos:
            self.logger.info(f"Joint mask enabled: {len(self.joint_mask_indices)}/{self.split_pos} gesture dimensions will be used")
            self.logger.info(f"Masked joints: {', '.join(joint_mask_names)}")
        else:
            self.logger.info(f"No joint mask applied - all {self.split_pos} gesture dimensions will be used")
        
        self.logger.info(f"Model output dimensions: {self.net_dim_pose} (gesture: {self.split_pos}, expression: {self.net_dim_pose - self.split_pos})")

        
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
        
        # Load delayed start seconds from config
        delayed_start_sec = self.config.get('co_speech_gestures', {}).get('delayed_start_sec', None)
        
        # Load duration timeout threshold from config
        # If elapsed time exceeds prescribed duration by this threshold, stop utterance
        self.duration_timeout_threshold = self.config.get('co_speech_gestures', {}).get('duration_timeout_threshold', 2.0)

        # Load blend to neutral duration from config
        self.blend_to_neutral_duration = self.config.get('co_speech_gestures', {}).get('blend_to_neutral_duration', 1.0)
        self.logger.info(f"Blend to neutral duration: {self.blend_to_neutral_duration}s")

        # Playback state management
        self.playback_state = PlaybackState.IDLE
        self.playback_state_lock = threading.Lock()
        
        # Blend window tracking
        self.blend_window: Optional[WaypointWindow] = None
        self.blend_start_time: Optional[float] = None
        self.blend_last_executed_waypoint_index: int = -1

        # Current utterance tracking (created once and reused)
        self.current_utterance: Utterance = Utterance(
            utterance_id=Utterance.PLACE_HOLDER_ID,  # Placeholder ID, will be updated when first chunk arrives
            sample_rate=self.audio_sr,
            gesture_fps=self.gesture_fps,
            window_size=self.window_size,
            window_step=self.window_step,
            generation_delayed_start_sec=delayed_start_sec
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
        
        # Track the last executed waypoint for monitoring and debugging
        self.last_executed_waypoint: Optional[GestureWaypoint] = None
        
        # Create prefilled neutral window for delayed start (if configured)
        self.prefill_neutral_window: Optional[WaypointWindow] = None
        if delayed_start_sec is not None and delayed_start_sec > 0:
            self.prefill_neutral_window = self._create_neutral_window()
            self.logger.info(f"Created neutral prefill window for delayed start ({delayed_start_sec}s)")
        
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

    def _create_neutral_window(self) -> WaypointWindow:
        """
        Create a prefilled window with neutral positions for all gestures.
        
        This is used when generation_delayed_start_sec > 0 to provide initial playable content
        during the delayed start period. All waypoints use the neutral position defined in
        self.neutral_position (which includes masked joint customizations).
        
        Returns:
            WaypointWindow with window_step execution waypoints + overlap_len context waypoints,
            all filled with neutral positions. Timestamps are NOT set here; they will be set
            in initialize_new_utterance when start_time is known.
        """
        execution_waypoints = []
        context_waypoints = []
        
        # Create execution waypoints (window_step frames) with neutral positions
        for i in range(self.window_step):
            waypoint = GestureWaypoint(
                waypoint_index=i,  # Temporary, will be updated in initialize_new_utterance
                timestamp=0.0,     # Temporary, will be updated in initialize_new_utterance
                gesture_data=self.neutral_position.copy(),
                is_for_execution=True
            )
            execution_waypoints.append(waypoint)
        
        # Create context waypoints (overlap_len frames) with neutral positions
        for i in range(self.overlap_len):
            waypoint = GestureWaypoint(
                waypoint_index=self.window_step + i,  # Temporary, will be updated in initialize_new_utterance
                timestamp=0.0,                        # Temporary, will be updated in initialize_new_utterance
                gesture_data=self.neutral_position.copy(),
                is_for_execution=False
            )
            context_waypoints.append(waypoint)
        
        # Create window (window_index will be 0 for the prefill)
        window = WaypointWindow(
            window_index=0,
            execution_waypoints=execution_waypoints,
            context_waypoints=context_waypoints
        )
        

        # Set proper timestamps for all waypoints
        # Frame indices start at 0 for the prefilled window, timestamps are relative to utterance start
        for i, waypoint in enumerate(window.execution_waypoints):
            waypoint.waypoint_index = i
            waypoint.timestamp = i / self.gesture_fps
        
        # Set timestamps for context waypoints
        for i, waypoint in enumerate(window.context_waypoints):
            waypoint.waypoint_index = self.window_step + i
            waypoint.timestamp = (self.window_step + i) / self.gesture_fps

        return window

    def _create_blend_to_neutral_window(self, last_waypoint: GestureWaypoint, blend_duration_sec: float) -> WaypointWindow:
        """
        Create a window that blends from the last executed waypoint to neutral position.
        
        This window linearly interpolates the masked joint angles from the last executed
        waypoint to the neutral position over the specified duration.
        
        Args:
            last_waypoint: The last executed waypoint to blend from
            blend_duration_sec: Duration of the blend in seconds
            
        Returns:
            WaypointWindow containing the blend trajectory
        """
        # Calculate number of frames for the blend based on gesture FPS
        num_frames = max(1, int(blend_duration_sec * self.gesture_fps))
        
        # Get start and end positions
        start_pose = last_waypoint.gesture_data.copy()
        end_pose = self.neutral_position.copy()
        
        # Create waypoints with linear interpolation
        execution_waypoints = []
        for i in range(num_frames):
            # Linear interpolation factor (0 at start, 1 at end)
            alpha = (i + 1) / num_frames
            
            # Interpolate between start and end
            blended_pose = start_pose * (1 - alpha) + end_pose * alpha
            
            waypoint = GestureWaypoint(
                waypoint_index=i,
                timestamp=i / self.gesture_fps,  # Relative to blend start
                gesture_data=blended_pose,
                is_for_execution=True
            )
            execution_waypoints.append(waypoint)
        
        # No context waypoints needed for blend window (this is the final window)
        window = WaypointWindow(
            window_index=0,  # Blend window is standalone
            execution_waypoints=execution_waypoints,
            context_waypoints=[]
        )
        
        return window

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
        self.logger.info("="*60)
        self.logger.info("STREAMING MODE: Real-time generation")
        if DiffSHEGRealtimeWrapper.USE_CONSTRAINED_FEATURES:
            self.logger.info(f"CONSTRAINED FEATURES: Enabled (duration={DiffSHEGRealtimeWrapper.AUDIO_DUR_FOR_FEATURES}s)")
        else:
            self.logger.info("CONSTRAINED FEATURES: Disabled (using full audio context)")
        self.logger.info("="*60)
        
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
        
        This clears the current utterance content, stopped history, and playback state.
        Useful for starting fresh or cleaning up after a session ends.
        """        
        with self.utterance_lock:
            self.current_utterance.clear()
            self.stopped_utterances.clear()
            self.last_executed_waypoint = None
        
        with self.playback_state_lock:
            self.playback_state = PlaybackState.IDLE
            self.blend_window = None
            self.blend_start_time = None
            self.blend_last_executed_waypoint_index = -1

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
            chunk_index: Position of this chunk within the utterance (starts from 0, note that this idx is the global index and won't reset with the end of an utterance)
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
                
                ## Stop the old utterance if it exists (this will initiate blend or transition to idle)
                self.stop_current_utterance(will_lock=False)
                
                ## Initialize the new utterance with ID and start time
                self.current_utterance.initialize_new_utterance(
                    utterance_id=utterance_id,
                    start_time=curren_time,
                    prefill_neutral_window=self.prefill_neutral_window
                )
                self.logger.info(f"Utterance object changed to new id={utterance_id}, playback should have started.")
                
                ## Update playback state to IN_UTTERANCE
                with self.playback_state_lock:
                    # Cancel any ongoing blend
                    if self.playback_state == PlaybackState.BLENDING_TO_NEUTRAL:
                        self.logger.info("New utterance started, cancelling ongoing blend")
                        self.blend_window = None
                        self.blend_start_time = None
                        self.blend_last_executed_waypoint_index = -1
                    
                    self.playback_state = PlaybackState.IN_UTTERANCE
                    self.logger.debug(f"Playback state transitioned to IN_UTTERANCE for utterance {utterance_id}")
                        
            # Add audio samples to utterance
            audio_samples_before = self.current_utterance.get_total_samples()
            self.current_utterance.add_audio_samples(audio_data)
            audio_samples_after = self.current_utterance.get_total_samples()
            
            # ## Avoid too frequent logging
            # self.logger.debug(f"Utterance {utterance_id} chunk {chunk_index}: added {audio_samples_after - audio_samples_before} samples (total: {audio_samples_after})")
    
    def stop_current_utterance(self, will_lock: bool):
        """
        Clear the current utterance content and initiate blending to neutral.
        
        Since gesture generation is faster than realtime, it's safe to simply
        clear the current utterance and all its waypoints when stopped.
        The system will start fresh with the next utterance using the same object.
        
        This also prevents late-arriving chunks for this utterance from being
        processed by tracking stopped utterance IDs in a set.
        
        After stopping the utterance, this method initiates a blend to neutral position
        if there was a last executed waypoint. Blend is only initiated once per utterance,
        even if this method is called multiple times (e.g., timeout followed by cancellation).
        
        Thread-safe: Can be called from multiple sources without duplicate blend initiation.
        """
        if will_lock:
            self.utterance_lock.acquire()

        try:
            # Add to stopped set to reject any late-arriving chunks
            if self.current_utterance.utterance_id != Utterance.PLACE_HOLDER_ID:
                utterance_id = self.current_utterance.utterance_id
                
                # Check if this utterance was already stopped
                if utterance_id in self.stopped_utterances:
                    self.logger.debug(f"Utterance {utterance_id} already stopped, skipping duplicate stop")
                    return
                
                # Mark as stopped
                self.stopped_utterances.add(utterance_id)
                
                # Clear the utterance content but keep the object alive
                total_samples = self.current_utterance.get_total_samples()
                self.logger.debug(f"Stop utterance {utterance_id} (had {total_samples} samples)")

                # Clear all content for reuse
                self.current_utterance.clear()
                
                # Initiate blend to neutral if we have a last waypoint
                # The _initiate_blend_to_neutral method will check the current state
                # to ensure blend is only started when transitioning from IN_UTTERANCE
                self._initiate_blend_to_neutral(self.last_executed_waypoint)
                
        except Exception as e:
            self.logger.error(f'Failed to stop utterance with exception: {type(e).__name__}: {e}')
            self.logger.error(f'Traceback:\n{traceback.format_exc()}')

        finally:
            if will_lock:
                self.utterance_lock.release()

    def register_utterance_end(self, utterance_duration_sec: float = 0.0, msg_idx: int = -1, will_lock: bool = False):
        """
        Register the end of the current utterance with duration information.
        
        This method is called when the audio LLM completes an utterance and provides
        the total duration. It updates the utterance duration and then clears the 
        utterance content for reuse.
        
        Args:
            utterance_duration_sec: Total duration of the utterance in seconds
            msg_idx: Message index identifying the audio message (for logging/tracing)
            will_lock: Whether to acquire the utterance lock before accessing the utterance
        """
        if will_lock:
            self.utterance_lock.acquire()

        try:
            # Add to stopped set to reject any late-arriving chunks
            if self.current_utterance.utterance_id != Utterance.PLACE_HOLDER_ID:
                
                # Update the utterance duration
                self.current_utterance.total_duration = utterance_duration_sec
                
                self.logger.info(f"Register utterance {self.current_utterance.utterance_id} (msg_idx={msg_idx}) as completely sliced, total duration: {utterance_duration_sec:.3f}s.")
                                
        except Exception as e:
            self.logger.error(f'Failed to register utterance end with exception: {type(e).__name__}: {e}')
            self.logger.error(f'Traceback:\n{traceback.format_exc()}')

        finally:
            if will_lock:
                self.utterance_lock.release()

    def _initiate_blend_to_neutral(self, last_waypoint: Optional[GestureWaypoint]):
        """
        Initiate blending to neutral position from the last executed waypoint.
        
        This method should be called when transitioning out of an utterance.
        It creates a blend window and updates the playback state.
        
        State transition guard: Only initiates blend when currently IN_UTTERANCE.
        This prevents duplicate blend initiation if stop_current_utterance is called
        multiple times (e.g., timeout followed by cancellation).
        
        Args:
            last_waypoint: The last executed waypoint to blend from, or None if no waypoint was executed
        """
        with self.playback_state_lock:
            # Check current state - only initiate blend if we're currently IN_UTTERANCE
            if self.playback_state != PlaybackState.IN_UTTERANCE:
                self.logger.debug(f"Skipping blend initiation - not in IN_UTTERANCE state (current: {self.playback_state.value})")
                return
            
            # Only initiate blend if we have a last waypoint and blend duration > 0
            if last_waypoint is not None and self.blend_to_neutral_duration > 0:
                # Create blend window
                self.blend_window = self._create_blend_to_neutral_window(
                    last_waypoint=last_waypoint,
                    blend_duration_sec=self.blend_to_neutral_duration
                )
                self.blend_start_time = time.time()
                self.blend_last_executed_waypoint_index = -1
                
                # Update state
                self.playback_state = PlaybackState.BLENDING_TO_NEUTRAL
                self.logger.info(f"Initiated blend to neutral (duration={self.blend_to_neutral_duration}s, {len(self.blend_window.execution_waypoints)} frames)")
            else:
                # No blending needed, go directly to idle
                self.blend_window = None
                self.blend_start_time = None
                self.blend_last_executed_waypoint_index = -1
                self.playback_state = PlaybackState.IDLE
                self.logger.debug("No blend needed, transitioning directly to IDLE")

    '''
    Track utterance playback and manage gesture playback
    '''
    def _playback_managing_loop(self):
        """
        Thread 1: Playback Managing Thread with State Machine
        
        This thread manages gesture playback through three states:
        1. IN_UTTERANCE: Executing waypoints for an active utterance
        2. BLENDING_TO_NEUTRAL: Executing blend window to return to neutral
        3. IDLE: At neutral position, nothing to execute
        
        State transitions:
        - IDLE -> IN_UTTERANCE: When first audio chunk arrives
        - IN_UTTERANCE -> BLENDING_TO_NEUTRAL: When utterance stops/ends/times out
        - IN_UTTERANCE -> IN_UTTERANCE: When new utterance starts (cancels previous)
        - BLENDING_TO_NEUTRAL -> IDLE: When blend completes
        - BLENDING_TO_NEUTRAL -> IN_UTTERANCE: When new utterance interrupts blend
        
        Waypoint execution:
        - Gestures are generated as waypoints at 15 FPS (one every ~66.67ms)
        - This thread checks every 10ms for waypoints to execute
        - At most 1 waypoint per 10ms interval (since waypoints are 66.67ms apart)
        """
        interval_duration = 0.01  # 10ms target interval
        
        while self.running:
            iteration_start_time = time.time()
            
            with self.playback_state_lock:
                current_state = self.playback_state
            
            # State-specific behavior
            if current_state == PlaybackState.IDLE:
                # Nothing to do in IDLE state
                pass
                
            elif current_state == PlaybackState.IN_UTTERANCE:
                # Handle utterance playback
                with self.utterance_lock:
                    # Skip if no valid utterance (placeholder ID)
                    if (
                        self.current_utterance.utterance_id == Utterance.PLACE_HOLDER_ID 
                        or self.current_utterance.start_time == Utterance.PLACE_HOLDER_TIMESTAMP):# Skip if playback hasn't started yet (no chunk 0 received)
                        pass
                    else:
                        # Calculate current playback position (time relative to utterance start)
                        elapsed_time = iteration_start_time - self.current_utterance.start_time
                        
                        # Check if utterance has exceeded its prescribed duration by threshold amount
                        if ( 
                            self.current_utterance.total_duration is not None 
                            and self.duration_timeout_threshold > 0 
                            and elapsed_time > self.current_utterance.total_duration + self.duration_timeout_threshold):
                            self.logger.info(
                                f"Utterance {self.current_utterance.utterance_id} exceeded duration by {elapsed_time - self.current_utterance.total_duration:.3f}s "
                                f"(duration={self.current_utterance.total_duration:.3f}s, threshold={self.duration_timeout_threshold:.3f}s). Stopping."
                            )
                            self.stop_current_utterance(will_lock=False)
                        else:
                            # Check for waypoint to execute in the next 10ms interval
                            waypoint = self.current_utterance.get_waypoint_to_execute_for_interval(
                                current_time=elapsed_time,
                                interval_duration=interval_duration
                            )
                            if waypoint is not None:
                                # # Execute waypoint gesture
                                # self.logger.debug(f"Utterance {self.current_utterance.utterance_id} executing waypoint {waypoint.waypoint_index} at t={elapsed_time:.3f}s (timestamp={waypoint.timestamp:.3f}s)")
                                # Store the last executed waypoint for monitoring and debugging
                                self.last_executed_waypoint = waypoint
                                # Call the waypoint callback if provided
                                if self.waypoint_callback is not None:
                                    self.waypoint_callback(waypoint)
                            
            elif current_state == PlaybackState.BLENDING_TO_NEUTRAL:
                # Handle blend window playback
                with self.playback_state_lock:
                    if self.blend_window is None or self.blend_start_time is None:
                        # Blend was cancelled or completed, transition to idle
                        self.playback_state = PlaybackState.IDLE
                        self.logger.debug("Blend window cleared, transitioning to IDLE")
                    else:
                        # Calculate elapsed time since blend start
                        blend_elapsed = iteration_start_time - self.blend_start_time
                        
                        # Find waypoint to execute in this interval
                        waypoint_to_execute = None
                        for i in range(self.blend_last_executed_waypoint_index + 1, 
                                      len(self.blend_window.execution_waypoints)):
                            waypoint = self.blend_window.execution_waypoints[i]
                            # Check if waypoint falls within current interval
                            if waypoint.timestamp <= blend_elapsed + interval_duration:
                                waypoint_to_execute = waypoint
                                self.blend_last_executed_waypoint_index = i
                            else:
                                break
                        
                        # Execute waypoint if found
                        if waypoint_to_execute is not None:
                            self.logger.debug(f"Blend executing waypoint {waypoint_to_execute.waypoint_index} at t={blend_elapsed:.3f}s")
                            # Store as last executed waypoint
                            self.last_executed_waypoint = waypoint_to_execute
                            # Call callback
                            if self.waypoint_callback is not None:
                                self.waypoint_callback(waypoint_to_execute)
                        
                        # Check if blend is complete
                        if self.blend_last_executed_waypoint_index >= len(self.blend_window.execution_waypoints) - 1:
                            self.logger.info("Blend to neutral completed, transitioning to IDLE")
                            self.blend_window = None
                            self.blend_start_time = None
                            self.blend_last_executed_waypoint_index = -1
                            self.playback_state = PlaybackState.IDLE

            # Sleep for the remaining time to maintain 10ms interval
            elapsed = time.time() - iteration_start_time
            sleep_time = max(0, interval_duration - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    

    '''
    Sanity check mode: Non-streaming batch generation
    '''
    

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
    

    '''
    Gesture generation scheduling
    '''

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
                
                # Variables to track if this is a tail generation with zero-padding
                is_tail_generation = False
                tail_audio_samples = 0  # How many genuine audio samples in tail
                
                if available_samples < self.current_utterance.next_window_end_sample:
                    # Not enough samples for a full window
                    # Check if utterance is complete (total_duration has been set)
                    if self.current_utterance.total_duration is None:
                        # Utterance not complete yet, more audio may arrive - keep waiting
                        continue
                    else:
                        # Utterance is complete, but has insufficient tail audio
                        # Generate gestures for the tail with zero-padding
                        if available_samples <= self.current_utterance.next_window_start_sample:
                            # No new audio samples in this window, skip it
                            continue
                        
                        is_tail_generation = True
                        tail_audio_samples = available_samples - self.current_utterance.next_window_start_sample
                        self.logger.info(
                            f"Utterance {self.current_utterance.utterance_id} tail generation: "
                            f"available_samples={available_samples}, "
                            f"window_start={self.current_utterance.next_window_start_sample}, "
                            f"tail_samples={tail_audio_samples}"
                        )
                
                # We have enough samples (or this is tail generation), prepare to generate
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
                    
                    if is_tail_generation:
                        # For tail generation, get available audio and zero-pad
                        audio_snapshot_full = self.current_utterance.get_audio_window(audio_start_sample, available_samples)
                        # Zero-pad to expected length
                        expected_bytes = (window_end_sample - audio_start_sample) * self.current_utterance.bytes_per_sample
                        padding_needed = expected_bytes - len(audio_snapshot_full)
                        if padding_needed > 0:
                            audio_snapshot_full += b'\x00' * padding_needed
                    else:
                        audio_snapshot_full = self.current_utterance.get_audio_window(audio_start_sample, window_end_sample)
                    
                    window_duration = (window_end_sample - window_start_sample) / sample_rate
                    constrained_duration = (window_end_sample - audio_start_sample) / sample_rate
                    self.logger.debug(
                        f"Utterance {utterance_id} generation triggered (CONSTRAINED{' - TAIL' if is_tail_generation else ''}): "
                        f"window [{window_start_sample}-{window_end_sample}] ({window_duration:.3f}s), "
                        f"audio_context [{audio_start_sample}-{window_end_sample}] ({constrained_duration:.3f}s), "
                        f"available_samples={available_samples}"
                    )
                else:
                    # Default mode: Snapshot ALL audio from start to window end (not just the window)
                    # This is needed because mel/HuBERT features must be computed over full context
                    # Following official code pattern: compute features for entire audio, then window them
                    audio_start_sample = 0
                    
                    if is_tail_generation:
                        # For tail generation, get available audio and zero-pad
                        audio_snapshot_full = self.current_utterance.get_audio_window(0, available_samples)
                        # Zero-pad to expected length
                        expected_bytes = window_end_sample * self.current_utterance.bytes_per_sample
                        padding_needed = expected_bytes - len(audio_snapshot_full)
                        if padding_needed > 0:
                            audio_snapshot_full += b'\x00' * padding_needed
                    else:
                        audio_snapshot_full = self.current_utterance.get_audio_window(0, window_end_sample)
                    
                    window_duration = (window_end_sample - window_start_sample) / sample_rate
                    self.logger.debug(
                        f"Utterance {utterance_id} generation triggered{' (TAIL)' if is_tail_generation else ''}: "
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
                
                # Calculate how many valid gesture frames correspond to genuine audio
                max_valid_frames = None
                if is_tail_generation:
                    # Convert tail audio samples to gesture frames
                    # Frames start from window_start_sample, and we have tail_audio_samples of valid audio
                    max_valid_frames = int(tail_audio_samples / sample_rate * gesture_fps)
                    self.logger.info(
                        f"Utterance {utterance_id} tail generation: "
                        f"tail_audio_samples={tail_audio_samples}, max_valid_frames={max_valid_frames}"
                    )
                
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
                    audio_context_start_sample=audio_start_sample,  # Pass context start for proper windowing
                    max_valid_frames=max_valid_frames  # Pass max valid frames for tail filtering
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
                    
                    # Update window indices for next generation (only if not tail generation)
                    # For tail generation, this was the last window, so no need to update indices
                    if max_valid_frames is None:
                        prev_window_end = self.current_utterance.next_window_end_sample
                        self.current_utterance.update_window_indices()
                        self.logger.debug(f"Utterance {utterance_id} window updated: next_window=[{self.current_utterance.next_window_start_sample}-{self.current_utterance.next_window_end_sample}] (step={self.current_utterance.next_window_start_sample - (prev_window_end - (self.current_utterance.next_window_end_sample - self.current_utterance.next_window_start_sample))} samples)")
                    else:
                        self.logger.info(f"Utterance {utterance_id} tail generation complete - no more windows to generate")

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
        audio_context_start_sample: int = 0,
        max_valid_frames: Optional[int] = None
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
            max_valid_frames: Optional maximum number of valid gesture frames to keep for tail generation.
                            When set, only the first max_valid_frames execution waypoints are kept,
                            as the remaining frames correspond to zero-padded audio.
        
        Returns:
            WaypointWindow containing execution waypoints and context waypoints
        """
        if len(audio_bytes_truncated) == 0 and precomputed_mel is None:
            return None

        if DO_AUD_NORMALIZATION:
            audio_bytes_truncated = normalize_audio_direct(audio_bytes_truncated)

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
        
        # Person ID (use configurable speaker_id)
        p_id = torch.ones((1, 1), device=self.device) * (self.speaker_id - 1)
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
        
        # Determine how many execution frames to create based on max_valid_frames
        num_execution_frames = self.window_step
        if max_valid_frames is not None:
            # For tail generation, only keep frames corresponding to genuine audio
            num_execution_frames = min(max_valid_frames, self.window_step)
            if num_execution_frames < self.window_step:
                self.logger.info(
                    f"Utterance {utterance_id} tail generation: "
                    f"keeping only {num_execution_frames}/{self.window_step} execution waypoints "
                    f"(frames with genuine audio)"
                )
        
        # Create execution waypoints (first num_execution_frames frames)
        for i in range(num_execution_frames):
            frame_index = window_start_frame + i
            timestamp = frame_index / gesture_fps
            
            waypoint = GestureWaypoint(
                waypoint_index=frame_index,
                timestamp=timestamp,
                gesture_data=outputs_np[i],  # [C,] - single frame
                is_for_execution=True
            )
            
            # Apply joint mask right after generation for consistency
            if self.joint_mask_indices and len(self.joint_mask_indices) < self.split_pos:
                waypoint = apply_joint_mask_to_waypoint(
                    waypoint,
                    self.joint_mask_indices,
                    split_pos=self.split_pos,
                    # custom_neural_positions_for_masked = self.neutral_position,
                )
            
            execution_waypoints.append(waypoint)
        
        # Create context waypoints (last overlap_len frames) - only if not tail generation
        # For tail generation with filtered frames, we don't create context waypoints
        # since there's no next window to use them
        if self.overlap_len > 0 and max_valid_frames is None:
            for i in range(self.overlap_len):
                frame_index = window_start_frame + self.window_step + i
                timestamp = frame_index / gesture_fps
                
                waypoint = GestureWaypoint(
                    waypoint_index=frame_index,
                    timestamp=timestamp,
                    gesture_data=outputs_np[self.window_step + i],  # [C,] - single frame
                    is_for_execution=False  # Context only, not for execution
                )
                
                # Apply joint mask to context waypoints as well for consistency
                if self.joint_mask_indices and len(self.joint_mask_indices) < self.split_pos:
                    waypoint = apply_joint_mask_to_waypoint(
                        waypoint,
                        self.joint_mask_indices,
                        split_pos=self.split_pos,
                        # custom_neural_positions_for_masked = self.neutral_position,
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
            + (f" (TAIL - filtered to {num_execution_frames} frames)" if max_valid_frames is not None else "")
        )
        return window

    
