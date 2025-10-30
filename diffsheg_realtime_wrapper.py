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

# Add parent directory to path to import logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger.logger import setup_logger

# Import for reference pipeline (only needed when SANITY_CHECK_USE_REFERENCE_PIPELINE is enabled)
try:
    from trainers.ddpm_beat_trainer import get_hubert_from_16k_speech_long
except ImportError:
    get_hubert_from_16k_speech_long = None


class Utterance:
    """Tracks an ongoing or completed utterance with audio stored as concatenated samples."""
    
    def __init__(
        self, 
        utterance_id: int,
        sample_rate: int,
        gesture_fps: int,
        window_size: int,
        window_step: int,
        gesture_waypoints: 'GestureWaypoints'
    ):
        self.utterance_id = utterance_id
        self.start_time: Optional[float] = None
        self.last_chunk_received_time: float = None # Track when last chunk arrived
        self.gesture_waypoints = gesture_waypoints  # Gesture waypoints for this utterance
        
        # Audio storage: concatenated samples instead of chunks
        self.sample_rate = sample_rate
        self.audio_samples: bytearray = bytearray()  # Concatenated raw audio samples (s16le encoded)
        self.bytes_per_sample = 2  # s16le encoding uses 2 bytes per sample
        
        # Generation state: tracks which audio window to generate next (in terms of sample indices)
        self.gesture_fps = gesture_fps
        self.window_size = window_size  # Number of frames per window
        self.window_step = window_step  # Number of non-overlapping frames per window
        
        # Initialize window indices - start from sample 0 (beginning of utterance)
        window_duration_samples = int((window_size / gesture_fps) * sample_rate)
        
        self.next_window_start_sample: int = 0
        self.next_window_end_sample: int = window_duration_samples
    
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
        
        self.audio_samples.extend(audio_bytes)
    
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
        
        # Clamp to available data
        end_byte = min(end_byte, len(self.audio_samples))
        
        if start_byte >= len(self.audio_samples):
            return bytes()
        
        return bytes(self.audio_samples[start_byte:end_byte])
    
    def update_window_indices(self):
        """
        Update window indices for the next generation window.
        Advances by window_step frames worth of samples.
        """
        step_duration_samples = int((self.window_step / self.gesture_fps) * self.sample_rate)
        window_duration_samples = int((self.window_size / self.gesture_fps) * self.sample_rate)
        
        self.next_window_start_sample += step_duration_samples
        self.next_window_end_sample = self.next_window_start_sample + window_duration_samples


@dataclass
class GestureWaypoint:
    """Represents a single gesture waypoint at a specific timestamp."""
    waypoint_index: int  # Sequential index of this waypoint (0, 1, 2, ...)
    timestamp: float  # Time in seconds from utterance start when this waypoint should be executed
    gesture_data: np.ndarray  # Gesture pose vector of shape (C,)


class GestureWaypoints:
    """
    Manages gesture waypoints for an utterance.
    
    Waypoints are generated at gesture_fps (15 FPS for BEAT), meaning one waypoint
    every ~66.67ms. The playback thread checks every 10ms if any waypoint should
    be executed in the upcoming interval.
    """
    
    def __init__(self, gesture_fps: int = 15):
        self.gesture_fps = gesture_fps
        self.waypoints: List[GestureWaypoint] = []
        self.last_executed_index: int = -1  # Track which waypoint was last executed
        self.lock = threading.Lock()
    
    def add_waypoints(self, waypoints: List[GestureWaypoint]):
        """Add new waypoints (from generation thread)."""
        with self.lock:
            self.waypoints.extend(waypoints)
            # Keep waypoints sorted by timestamp
            self.waypoints.sort(key=lambda w: w.timestamp)
    
    def get_waypoint_for_interval(self, current_time: float, interval_duration: float = 0.01) -> Optional[GestureWaypoint]:
        """
        Find the waypoint that should be executed in the next interval.
        
        Args:
            current_time: Current playback time in seconds from utterance start
            interval_duration: Duration of the upcoming interval (default 10ms = 0.01s)
            
        Returns:
            The waypoint to execute, or None if no waypoint falls in this interval.
            At most one waypoint per interval since waypoints are at 15 FPS (~66.67ms apart).
        """
        with self.lock:
            if not self.waypoints:
                return None
            
            interval_end = current_time + interval_duration
            
            # Search from the last executed waypoint onwards
            search_start = self.last_executed_index + 1
            
            for i in range(search_start, len(self.waypoints)):
                waypoint = self.waypoints[i]
                
                # Check if this waypoint falls within the upcoming interval
                if current_time <= waypoint.timestamp < interval_end:
                    self.last_executed_index = i
                    return waypoint
                
                # If waypoint is beyond the interval, we can stop searching
                if waypoint.timestamp >= interval_end:
                    break
            
            return None
    
    def get_total_waypoints(self):
        with self.lock:
            return len(self.waypoints)

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
    """
    
    # Global flag for sanity check mode
    NON_STREAMING_SANITY_CHECK = False
    
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
        
        # Current utterance tracking (only keep the latest)
        self.current_utterance: Optional[Utterance] = None
        self.utterance_lock = threading.Lock()
        
        # Track cancelled/timed-out utterances to reject late-arriving chunks
        self.cancelled_utterances: set = set()  # Set of utterance_ids that have been cancelled or timed out
        
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
        
        # DiffSHEG configuration
        self.window_size = opt.n_poses  # e.g., 34 frames
        self.overlap_len = opt.overlap_len  # e.g., 4 frames
        self.window_step = self.window_size - self.overlap_len
        
        # Compute frames per audio second
        # DiffSHEG uses 15 FPS for BEAT dataset
        self.gesture_fps = 15
        
        self.logger.info(f"DiffSHEG parameters: window_size={self.window_size}, overlap={self.overlap_len}, step={self.window_step}, fps={self.gesture_fps}")

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


    def start(self):
        """Start the wrapper threads."""
        self.running = True
        
        # Log mode information
        if DiffSHEGRealtimeWrapper.NON_STREAMING_SANITY_CHECK:
            if DiffSHEGRealtimeWrapper.SANITY_CHECK_USE_REFERENCE_PIPELINE:
                self.logger.info("="*60)
                self.logger.info("SANITY CHECK MODE: REFERENCE PIPELINE")
                self.logger.info("Using EXACT same code as official runner.py")
                self.logger.info("="*60)
            else:
                self.logger.info("="*60)
                self.logger.info("SANITY CHECK MODE: NON-STREAMING")
                self.logger.info("Using wrapper's own generation pipeline")
                self.logger.info("="*60)
        else:
            self.logger.info("="*60)
            self.logger.info("STREAMING MODE: Real-time generation")
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
        
        This clears the current utterance and cancellation history.
        Useful for starting fresh or cleaning up after a session ends.
        """
        with self.utterance_lock:
            old_utterance_id = self.current_utterance.utterance_id if self.current_utterance else None
            self.current_utterance = None
            self.cancelled_utterances.clear()
        
        if old_utterance_id is not None:
            self.logger.info(f"Context reset: cleared utterance {old_utterance_id} and cancellation history")
    
    '''
    Utterance lifecycle methods
    '''

    def add_audio_chunk(self, utterance_id: int, chunk_index: int, audio_data: list, 
                        duration: float):
        """
        Add a new audio chunk from the dialogue system.
        Playback automatically starts when the first chunk (chunk_index=0) arrives.
        Only keeps the latest utterance - previous utterances are discarded.
        
        Args:
            utterance_id: Unique identifier for the utterance (msg_idx in your system)
            chunk_index: Position of this chunk within the utterance (starts from 0)
            audio_data: Raw audio data (list of integers)
            duration: Duration of the chunk in seconds (not used internally, kept for API compatibility)
        """
        # Reject chunks for cancelled/timed-out utterances
        if utterance_id in self.cancelled_utterances:
            # Silently ignore - this is expected for late-arriving chunks after cancellation
            return
        
        curren_time = time.time()

        with self.utterance_lock:
            # Double-check after acquiring lock (in case it was cancelled while we were creating the chunk)
            if utterance_id in self.cancelled_utterances:
                return
            
            # If this is a new utterance, discard the old one
            if self.current_utterance is None or self.current_utterance.utterance_id != utterance_id:
                # Mark old utterance as cancelled if it exists
                if self.current_utterance is not None:
                    old_utterance_id = self.current_utterance.utterance_id
                    self.cancelled_utterances.add(old_utterance_id)
                    self.logger.info(f"Utterance {old_utterance_id} replaced by new utterance {utterance_id}")
                
                self.current_utterance = Utterance(
                    utterance_id=utterance_id,
                    sample_rate=self.audio_sr,
                    gesture_fps=self.gesture_fps,
                    window_size=self.window_size,
                    window_step=self.window_step,
                    gesture_waypoints=GestureWaypoints(gesture_fps=self.gesture_fps)
                )
                self.logger.info(f"New utterance created: id={utterance_id}")
            
            utterance = self.current_utterance
            
            # Update last chunk received time
            utterance.last_chunk_received_time = curren_time
            
            # Automatically start playback when first chunk arrives, we log the timestamp here
            if chunk_index == 0 and utterance.start_time is None:
                utterance.start_time = curren_time
                self.logger.info(f"Utterance {utterance_id} playback started at chunk 0")
            
            # Add audio samples to utterance
            audio_samples_before = utterance.get_total_samples()
            utterance.add_audio_samples(audio_data)
            audio_samples_after = utterance.get_total_samples()
            
            self.logger.debug(f"Utterance {utterance_id} chunk {chunk_index}: added {audio_samples_after - audio_samples_before} samples (total: {audio_samples_after})")
    
    def cancel_utterance(self, utterance_id: int):
        """
        Cancel an utterance and discard its audio chunks and generated gestures.
        
        Since gesture generation is faster than realtime, it's safe to simply
        discard the current utterance and all its waypoints when cancelled.
        The system will start fresh with the next utterance.
        
        This also prevents late-arriving chunks for this utterance from being
        processed by tracking cancelled utterance IDs in a set.
        
        Args:
            utterance_id: The utterance to cancel (msg_idx in your system)
        """
        with self.utterance_lock:
            # Add to cancelled set to reject any late-arriving chunks
            self.cancelled_utterances.add(utterance_id)
            
            # If this is the current utterance, discard it
            if self.current_utterance and self.current_utterance.utterance_id == utterance_id:
                # Simply discard everything - generation is faster than realtime
                # so we can regenerate from scratch for the next utterance
                total_samples = self.current_utterance.get_total_samples()
                total_waypoints = len(self.current_utterance.gesture_waypoints.waypoints) if self.current_utterance.gesture_waypoints else 0
                self.current_utterance = None
                self.logger.info(f"Utterance {utterance_id} cancelled (had {total_samples} samples, {total_waypoints} waypoints)")
            else:
                self.logger.info(f"Utterance {utterance_id} marked as cancelled (not current utterance)")
    
    def _cleanup_current_utterance(self):
        """
        Internal cleanup method called automatically by playback managing thread.
        Cleans up current utterance data after playback naturally ends.
        Also marks the utterance as cancelled to reject any late-arriving chunks.
        """
        with self.utterance_lock:
            if self.current_utterance is not None:
                utterance_id = self.current_utterance.utterance_id
                total_samples = self.current_utterance.get_total_samples()
                total_waypoints = len(self.current_utterance.gesture_waypoints.waypoints) if self.current_utterance.gesture_waypoints else 0
                duration_sec = total_samples / self.current_utterance.sample_rate
                
                # Mark as cancelled to reject late chunks
                self.cancelled_utterances.add(utterance_id)
    
                self.current_utterance = None
                
                self.logger.info(f"Utterance {utterance_id} auto-cleanup: playback ended naturally (duration={duration_sec:.2f}s, {total_waypoints} waypoints executed)")


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
                utterance = self.current_utterance
                
                if utterance is None:
                    continue
                
                # Skip if playback hasn't started yet (no chunk 0 received)
                if utterance.start_time is None:
                    continue
                
                # Calculate current playback position (time relative to utterance start)
                elapsed_time = iteration_start_time - utterance.start_time
                
                # Calculate total audio duration received so far
                total_samples = utterance.get_total_samples()
                total_audio_duration = total_samples / utterance.sample_rate
                

                # Auto-cleanup: if playback has passed all audio AND no new chunks for cleanup_timeout
                if elapsed_time > total_audio_duration:
                    time_since_last_chunk = iteration_start_time - utterance.last_chunk_received_time
                    if time_since_last_chunk > self.cleanup_timeout:
                        should_cleanup = True
                
                # Check for waypoint to execute in the next 10ms interval
                if utterance.gesture_waypoints is not None:
                    waypoint = utterance.gesture_waypoints.get_waypoint_for_interval(
                        current_time=elapsed_time,
                        interval_duration=interval_duration
                    )
                    if waypoint is not None:
                        # Execute waypoint gesture
                        self.logger.debug(f"Utterance {utterance.utterance_id} executing waypoint {waypoint.waypoint_index} at t={elapsed_time:.3f}s (timestamp={waypoint.timestamp:.3f}s)")
                        # Call the waypoint callback if provided
                        if self.waypoint_callback is not None:
                            self.waypoint_callback(waypoint)
            
            if should_cleanup:
                self._cleanup_current_utterance()
            
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
                utterance = self.current_utterance
                if utterance is None or utterance.start_time is None:
                    continue
            
            # Monitor for audio completion
            while self.running:
                time.sleep(0.05)
                
                with self.utterance_lock:
                    utterance = self.current_utterance
                    if utterance is None:
                        break
                    
                    # Check if we've stopped receiving chunks
                    if utterance.last_chunk_received_time is not None:
                        time_since_last_chunk = time.time() - utterance.last_chunk_received_time
                        if time_since_last_chunk >= audio_complete_threshold:
                            # Audio is complete!
                            utterance_id = utterance.utterance_id
                            total_samples = utterance.get_total_samples()
                            audio_duration = total_samples / utterance.sample_rate
                            
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
    
    def _sanity_check_generate_all(self):
        """Generate all windows for the current utterance using full audio context."""
        with self.utterance_lock:
            utterance = self.current_utterance
            if utterance is None:
                return
            
            utterance_id = utterance.utterance_id
            total_samples = utterance.get_total_samples()
            sample_rate = utterance.sample_rate
            gesture_fps = utterance.gesture_fps
            
            # Get FULL audio
            full_audio_bytes = utterance.get_audio_window(0, total_samples)
            
            self.logger.info(
                f"[SANITY CHECK] Generating all windows for utterance {utterance_id}: "
                f"total_samples={total_samples}, duration={total_samples/sample_rate:.2f}s"
            )
        
        # Generate all windows with audio truncation (simulating streaming)
        all_waypoints = []
        all_window_outputs = []  # Collect raw outputs for export
        overlap_context = None
        
        window_start_sample = 0
        window_end_sample = int((self.window_size / gesture_fps) * sample_rate)
        
        window_idx = 0
        while window_start_sample < total_samples:
            # Truncate audio to [0, window_end] (simulate streaming: pretend no future audio exists)
            audio_truncated = full_audio_bytes[:window_end_sample * 2]  # *2 because s16le (2 bytes/sample)
            
            self.logger.info(
                f"[SANITY CHECK] Generating window {window_idx}: "
                f"samples [{window_start_sample}-{window_end_sample}], "
                f"truncated_audio_samples={len(audio_truncated)//2}"
            )
            
            # Generate this window using TRUNCATED audio [0, window_end]
            waypoints = self._generate_gesture_window_from_audio(
                audio_truncated,  # CRITICAL: Pass truncated audio [0, window_end]
                window_start_sample,
                min(window_end_sample, total_samples),
                sample_rate,
                gesture_fps,
                overlap_context,
                utterance_id
            )
            
            all_waypoints.extend(waypoints)
            
            # Collect gesture data for export (only non-overlapping frames)
            # Each waypoint contains gesture_data with shape (C,) where C = net_dim_pose (192 for gesture+expression)
            window_gesture_data = np.array([wp.gesture_data for wp in waypoints])  # Shape: (window_step, C)
            all_window_outputs.append(window_gesture_data)
            
            # Update overlap context for next window
            if waypoints and self.overlap_len > 0:
                # Get last overlap_len waypoints as context for next window
                overlap_context = [wp.gesture_data.copy() for wp in waypoints[-self.overlap_len:]]
            
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
                utterance = self.current_utterance
                
                if utterance is None:
                    continue
                
                # Check if we have enough samples for the next window
                available_samples = utterance.get_total_samples()
                
                if available_samples < utterance.next_window_end_sample:
                    # Not enough samples yet, wait for more
                    continue
                
                # We have enough samples, prepare to generate
                should_generate = True
                utterance_id = utterance.utterance_id
                window_start_sample = utterance.next_window_start_sample
                window_end_sample = utterance.next_window_end_sample
                sample_rate = utterance.sample_rate
                gesture_fps = utterance.gesture_fps
                
                # CRITICAL FIX: Snapshot ALL audio from start to window end (not just the window)
                # This is needed because mel/HuBERT features must be computed over full context
                # Following official code pattern: compute features for entire audio, then window them
                audio_snapshot_full = utterance.get_audio_window(0, window_end_sample)
                
                window_duration = (window_end_sample - window_start_sample) / sample_rate
                self.logger.debug(f"Utterance {utterance_id} generation triggered: window [{window_start_sample}-{window_end_sample}] ({window_duration:.3f}s), available_samples={available_samples}, full_context_samples={window_end_sample}")
                
                # Snapshot overlap context if needed for smooth transitions
                # Frames are gesture poses indexed at gesture_fps (e.g., 15 FPS for BEAT)
                # Frame index = (sample_index / sample_rate) * gesture_fps
                # For example: at 16000 Hz and 15 FPS, 1 frame spans ~1067 samples
                window_start_frame = int(window_start_sample / sample_rate * gesture_fps)
                
                if self.overlap_len > 0 and window_start_frame > 0:
                    # We need the last overlap_len frames before window_start_frame for inpainting
                    if utterance.gesture_waypoints is not None:
                        prev_waypoints = []
                        with utterance.gesture_waypoints.lock:
                            for i in range(self.overlap_len):
                                prev_frame = window_start_frame - self.overlap_len + i
                                if prev_frame >= 0:
                                    # Find waypoint at this frame index
                                    for wp in utterance.gesture_waypoints.waypoints:
                                        if wp.waypoint_index == prev_frame:
                                            # Deep copy the gesture data to avoid reference issues
                                            prev_waypoints.append(wp.gesture_data.copy())
                                            break
                        
                        if len(prev_waypoints) == self.overlap_len:
                            overlap_context = prev_waypoints
            
            # Step 2: Generate gestures without holding the lock
            if should_generate and len(audio_snapshot_full) > 0:
                # Generate gestures for this window
                gen_start_time = time.time()
                waypoints = self._generate_gesture_window_from_audio(
                    audio_snapshot_full,  # Pass full audio context
                    window_start_sample,
                    window_end_sample,
                    sample_rate,
                    gesture_fps,
                    overlap_context,
                    utterance_id
                )
                gen_duration = time.time() - gen_start_time
                self.logger.debug(f"Utterance {utterance_id} generation completed: {len(waypoints)} waypoints in {gen_duration:.3f}s")
                
                # Step 3: Write waypoints back and update generation state
                with self.utterance_lock:
                    utterance = self.current_utterance
                    
                    # Check if this utterance is still current (not cancelled)
                    if utterance is None or utterance.utterance_id != utterance_id:
                        # Utterance was cancelled, discard waypoints
                        self.logger.debug(f"Utterance {utterance_id} was cancelled during generation, discarding {len(waypoints)} waypoints")
                        continue
                    
                    # Write waypoints
                    if waypoints and utterance.gesture_waypoints is not None:
                        utterance.gesture_waypoints.add_waypoints(waypoints)
                        self.logger.debug(f"Utterance {utterance_id} waypoints written: {len(waypoints)} waypoints added, total={utterance.gesture_waypoints.get_total_waypoints()}")
                    
                    # Update window indices for next generation
                    prev_window_end = utterance.next_window_end_sample
                    utterance.update_window_indices()
                    self.logger.debug(f"Utterance {utterance_id} window updated: next_window=[{utterance.next_window_start_sample}-{utterance.next_window_end_sample}] (step={utterance.next_window_start_sample - (prev_window_end - (utterance.next_window_end_sample - utterance.next_window_start_sample))} samples)")

    def _generate_gesture_window_from_audio(
        self, 
        audio_bytes_truncated: bytes,
        window_start_sample: int,
        window_end_sample: int,
        sample_rate: int,
        gesture_fps: int,
        overlap_context: Optional[List[np.ndarray]],
        utterance_id: int
    ) -> List[GestureWaypoint]:
        """
        Generate gestures for a single window using official DiffSHEG pipeline.
        
        This method implements the EXACT official generation code from 
        trainers/ddpm_beat_trainer.py::test_custom_aud(), adapted for single-window generation.
        
        KEY STRATEGY FOR STREAMING:
        - audio_bytes_truncated contains audio from [0, window_end_sample]
        - We compute mel/HuBERT features for this truncated audio (as if no future audio exists)
        - We then window these features to extract the current window portion
        - This ensures temporal consistency while enabling streaming
        
        Official DiffSHEG pipeline:
        1. Convert audio bytes to float32 (normalized to [-1, 1])
        2. Resample to 18kHz for mel spectrogram
        3. Extract mel: librosa.feature.melspectrogram(...) then mel[..., :-1]
        4. Extract HuBERT if enabled: get_hubert_from_16k_speech_long(...) then interpolate
        5. Window the features to get current window (window_size frames)
        6. Generate with trainer.generate_batch() using inpainting for overlap
        
        Args:
            audio_bytes_truncated: Raw audio bytes from [0, window_end_sample] (s16le encoded)
                                   This "pretends" that the audio only extends to window_end
            window_start_sample: Starting sample index of this window in the full utterance
            window_end_sample: Ending sample index of this window (where audio is truncated)
            sample_rate: Audio sample rate (e.g., 16000 Hz)
            gesture_fps: Gesture frame rate (e.g., 15 FPS for BEAT)
            overlap_context: Optional list of gesture data arrays from previous window's last frames
                           Used for smooth transitions via inpainting. Length should be overlap_len.
            utterance_id: ID of the utterance (for debugging/logging)
        
        Returns:
            List of generated waypoints (only the non-overlapping window_step frames)
        """
        if len(audio_bytes_truncated) == 0:
            return []
        
        # ===== STEP 1: Convert audio bytes to normalized float32 (official format) =====
        audio_int16 = np.frombuffer(audio_bytes_truncated, dtype=np.int16)
        if audio_int16.size == 0:
            return []
        # Normalize to [-1, 1] (official code does this for librosa processing)
        aud_ori = audio_int16.astype(np.float32) / 32768.0
        
        # ===== STEP 2: Resample to 18kHz (official mel extraction) =====
        aud = librosa.resample(aud_ori, orig_sr=sample_rate, target_sr=18000)
        
        # ===== STEP 3: Extract mel spectrogram (EXACT official code) =====
        mel = librosa.feature.melspectrogram(y=aud, sr=18000, hop_length=1200, n_mels=128)
        mel = mel[..., :-1]  # CRITICAL: Always remove last frame (official code line 1249)
        audio_emb = torch.from_numpy(np.swapaxes(mel, -1, -2))  # [T, 128]
        audio_emb = audio_emb.unsqueeze(0).to(self.device)  # [1, T, 128]
        
        # ===== STEP 4: Extract HuBERT features if enabled (EXACT official code) =====
        add_cond = {}
        if self.opt.expAddHubert or self.opt.addHubert:
            # Use official function from trainers.ddpm_beat_trainer
            hubert_feat = get_hubert_from_16k_speech_long(
                self.hubert_model,
                self.wav2vec2_processor,
                torch.from_numpy(aud_ori).unsqueeze(0).to(self.device),
                device=self.device
            )
            # Interpolate to match mel length
            hubert_feat = F.interpolate(
                hubert_feat.swapaxes(-1,-2).unsqueeze(0),
                size=audio_emb.shape[1],
                mode='linear',
                align_corners=True
            ).swapaxes(-1,-2)  # [1, T, 1024]
            add_cond["pretrain_aud_feat"] = hubert_feat.to(self.device)
        
        # ===== STEP 5: Window the features to get current window portion =====
        window_start_frame = int(round(window_start_sample / sample_rate * gesture_fps))
        
        # Extract window from full features (features span [0, window_end_sample])
        mel_window_start = window_start_frame
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
            
            if window_start_frame == 0:
                # First window: apply fix_very_first if enabled
                if self.opt.fix_very_first:
                    inpaint_dict['outpainting_mask'][:, :self.overlap_len, :] = True
                    inpaint_dict['gt'][:, :self.overlap_len, :] = 0  # Condition on neutral pose
            else:
                # Subsequent windows: use overlap context from previous window
                if overlap_context is not None and len(overlap_context) == self.overlap_len:
                    inpaint_dict['outpainting_mask'][:, :self.overlap_len, :] = True
                    prev_frames_np = np.stack(overlap_context).astype(np.float32)
                    prev_frames = torch.from_numpy(prev_frames_np).unsqueeze(0).to(self.device)
                    inpaint_dict['gt'][:, :self.overlap_len, :] = prev_frames
        
        # ===== STEP 8: Generate using official trainer method =====
        with torch.no_grad():
            outputs = self.model.generate_batch(
                audio_window, p_id, C, add_cond, inpaint_dict
            )
        
        outputs_np = outputs.cpu().numpy()[0]  # [window_size, C]
        
        # ===== STEP 9: Create waypoints (only non-overlapping window_step frames) =====
        waypoints = []
        for i in range(self.window_step):
            frame_index = window_start_frame + i
            timestamp = frame_index / gesture_fps
            
            waypoint = GestureWaypoint(
                waypoint_index=frame_index,
                timestamp=timestamp,
                gesture_data=outputs_np[i]  # [C,]
            )
            waypoints.append(waypoint)
        
        self.logger.debug(
            f"Utterance {utterance_id} window {window_start_frame}: "
            f"generated {len(waypoints)} waypoints"
        )
        return waypoints

    

# Example usage
if __name__ == "__main__":
    # This is a demonstration of how to use the wrapper
    # In practice, you would initialize this with your trained DiffSHEG model
    
    # Example: Initialize wrapper (pseudo-code)
    # wrapper = DiffSHEGRealtimeWrapper(
    #     diffsheg_model=your_trained_model,
    #     opt=your_opt_config,
    #     start_margin=0.5,
    #     audio_sr=16000,
    #     device="cuda"
    # )
    # wrapper.start()
    # 
    # # Add audio chunks as they arrive from your audio system
    # # Audio data can be: bytes, bytearray, list of integers (as from app.py), or numpy arrays
    # # Playback starts automatically when chunk_index=0 arrives
    # wrapper.add_audio_chunk(utterance_id=1, chunk_index=0, audio_data=chunk0, duration=0.1)  # Playback starts here
    # wrapper.add_audio_chunk(utterance_id=1, chunk_index=1, audio_data=chunk1, duration=0.1)
    # wrapper.add_audio_chunk(utterance_id=1, chunk_index=2, audio_data=chunk2, duration=0.1)
    # # ... chunks stop arriving naturally when audio generation finishes
    # # Cleanup happens automatically after playback ends (default 2 seconds after last audio)
    # 
    # # Cancel if interrupted (only explicit cancellation signal needed)
    # # wrapper.cancel_utterance(utterance_id=1)
    # 
    # # Cleanup
    # wrapper.stop()
    
    print("DiffSHEG Realtime Wrapper - See class documentation for usage")
