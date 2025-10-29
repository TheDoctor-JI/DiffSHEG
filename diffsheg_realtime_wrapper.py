"""
DiffSHEG Realtime Wrapper for GPT-4o Dialogue System Integration

This wrapper class manages the integration of DiffSHEG gesture generation with
a real-time dialogue system that receives audio chunks from GPT-4o-realtime.
"""

import threading
import time
import queue
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np
import torch
import librosa


class Utterance:
    """Tracks an ongoing or completed utterance with audio stored as concatenated samples."""
    
    def __init__(
        self, 
        utterance_id: int,
        start_margin: float,
        sample_rate: int,
        gesture_fps: int,
        window_size: int,
        window_step: int,
        gesture_waypoints: 'GestureWaypoints'
    ):
        self.utterance_id = utterance_id
        self.start_time: Optional[float] = None
        self.start_margin = start_margin  # Time offset (in seconds) from utterance start where gesture generation begins
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
        
        # Initialize window indices based on start_margin
        # First window starts at start_margin seconds into the utterance
        start_margin_samples = int(start_margin * sample_rate)
        window_duration_samples = int((window_size / gesture_fps) * sample_rate)
        
        self.next_window_start_sample: int = start_margin_samples
        self.next_window_end_sample: int = start_margin_samples + window_duration_samples
    
    def add_audio_samples(self, audio_data: np.ndarray):
        """
        Add audio samples to the utterance.
        
        Args:
            audio_data: Audio samples as numpy array (will be converted to s16le bytes)
        """
        # Convert to int16 and then to bytes
        if audio_data.dtype != np.int16:
            audio_int16 = (audio_data * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data
        
        audio_bytes = audio_int16.tobytes()
        self.audio_samples.extend(audio_bytes)
    
    def get_total_samples(self) -> int:
        """Get total number of audio samples accumulated."""
        return len(self.audio_samples) // self.bytes_per_sample
    
    def get_audio_window(self, start_sample: int, end_sample: int) -> np.ndarray:
        """
        Extract audio samples for a specific window.
        
        Args:
            start_sample: Starting sample index
            end_sample: Ending sample index (exclusive)
            
        Returns:
            Audio data as numpy array
        """
        start_byte = start_sample * self.bytes_per_sample
        end_byte = end_sample * self.bytes_per_sample
        
        # Clamp to available data
        end_byte = min(end_byte, len(self.audio_samples))
        
        if start_byte >= len(self.audio_samples):
            return np.array([], dtype=np.int16)
        
        window_bytes = bytes(self.audio_samples[start_byte:end_byte])
        audio_int16 = np.frombuffer(window_bytes, dtype=np.int16)
        
        # Convert to float32 in range [-1, 1]
        return audio_int16.astype(np.float32) / 32767.0
    
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
    

class DiffSHEGRealtimeWrapper:
    """
    Wrapper for integrating DiffSHEG with real-time dialogue systems.
    
    Features:
    - Tracks utterance lifecycle and audio chunks
    - Schedules gesture generation with configurable start margin
    - Manages two threads: playback monitoring and gesture generation
    - Generation thread uses snapshot-based approach for lock-free inference
    """
    
    def __init__(
        self,
        diffsheg_model,
        opt,
        default_start_margin: float = 0.5,  # Time offset from utterance start to begin gesture generation
        audio_sr: int = 16000,
        device: str = "cuda",
        cleanup_timeout: float = 2.0  # Seconds after playback ends to auto-cleanup
    ):
        """
        Initialize the wrapper.
        
        Args:
            diffsheg_model: The DiffSHEG trainer instance (DDPMTrainer_beat)
            opt: Configuration options
            default_start_margin: Time offset (in seconds) from utterance start where gesture 
                                 generation begins. For example, if start_margin=0.5, the first
                                 generation window will start at 0.5s into the utterance.
                                 This ensures gestures are ready before playback needs them,
                                 since generation is faster than realtime.
            audio_sr: Audio sample rate
            device: Computing device
            cleanup_timeout: Seconds to wait after playback ends before auto-cleanup
        """
        self.model = diffsheg_model
        self.opt = opt
        self.default_start_margin = default_start_margin
        self.audio_sr = audio_sr
        self.device = device
        self.cleanup_timeout = cleanup_timeout
        
        # Current utterance tracking (only keep the latest)
        self.current_utterance: Optional[Utterance] = None
        self.utterance_lock = threading.Lock()
        
        # Track cancelled/timed-out utterances to reject late-arriving chunks
        self.cancelled_utterances: set = set()  # Set of utterance_ids that have been cancelled or timed out
        
        # Threading
        self.running = False
        self.playback_managing_thread: Optional[threading.Thread] = None
        self.generation_thread: Optional[threading.Thread] = None
        
        # DiffSHEG configuration
        self.window_size = opt.n_poses  # e.g., 34 frames
        self.overlap_len = opt.overlap_len  # e.g., 4 frames
        self.window_step = self.window_size - self.overlap_len
        
        # Compute frames per audio second
        # DiffSHEG uses 15 FPS for BEAT dataset
        self.gesture_fps = 15
        
    def start(self):
        """Start the wrapper threads."""
        self.running = True
        self.playback_managing_thread = threading.Thread(target=self._playback_managing_loop, daemon=True)
        self.generation_thread = threading.Thread(target=self._generation_loop, daemon=True)
        self.playback_managing_thread.start()
        self.generation_thread.start()
        
    def stop(self):
        """Stop the wrapper threads."""
        self.running = False
        if self.playback_managing_thread:
            self.playback_managing_thread.join(timeout=2.0)
        if self.generation_thread:
            self.generation_thread.join(timeout=2.0)
    
    def reset_context(self):
        """
        Reset all state-related variables.
        
        This clears the current utterance and cancellation history.
        Useful for starting fresh or cleaning up after a session ends.
        """
        with self.utterance_lock:
            self.current_utterance = None
            self.cancelled_utterances.clear()
    
    '''
    Utterance lifecycle methods
    '''

    def add_audio_chunk(self, utterance_id: int, chunk_index: int, audio_data: np.ndarray, 
                        duration: float, start_margin: Optional[float] = None):
        """
        Add a new audio chunk from the dialogue system.
        Playback automatically starts when the first chunk (chunk_index=0) arrives.
        Only keeps the latest utterance - previous utterances are discarded.
        
        Args:
            utterance_id: Unique identifier for the utterance (msg_idx in your system)
            chunk_index: Position of this chunk within the utterance (starts from 0)
            audio_data: Raw audio data (numpy array)
            duration: Duration of the chunk in seconds
            start_margin: Optional time offset (in seconds) for when gesture generation starts.
                         If None, uses default_start_margin. The first generation window will
                         begin at this offset from the utterance start. For example, with
                         start_margin=0.5, generation begins at t=0.5s into the utterance.
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
                
                margin = start_margin if start_margin is not None else self.default_start_margin
                self.current_utterance = Utterance(
                    utterance_id=utterance_id,
                    start_margin=margin,
                    sample_rate=self.audio_sr,
                    gesture_fps=self.gesture_fps,
                    window_size=self.window_size,
                    window_step=self.window_step,
                    gesture_waypoints=GestureWaypoints(gesture_fps=self.gesture_fps)
                )
            
            utterance = self.current_utterance
            
            # Update last chunk received time
            utterance.last_chunk_received_time = curren_time
            
            # Automatically start playback when first chunk arrives, we log the timestamp here
            if chunk_index == 0 and utterance.start_time is None:
                utterance.start_time = curren_time
            
            # Add audio samples to utterance
            utterance.add_audio_samples(audio_data)
    
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
                self.current_utterance = None
    
    def _cleanup_current_utterance(self):
        """
        Internal cleanup method called automatically by playback managing thread.
        Cleans up current utterance data after playback naturally ends.
        Also marks the utterance as cancelled to reject any late-arriving chunks.
        """
        with self.utterance_lock:
            if self.current_utterance is not None:
                # Mark as cancelled to reject late chunks
                self.cancelled_utterances.add(self.current_utterance.utterance_id)
    
            self.current_utterance = None


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
                        # TODO: Send waypoint gesture to robot control system
                        # self.send_to_robot(waypoint.gesture_data)
                        pass
            
            if should_cleanup:
                self._cleanup_current_utterance()
            
            # Sleep for the remaining time to maintain 10ms interval
            elapsed = time.time() - iteration_start_time
            sleep_time = max(0, interval_duration - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    

    '''
    Gesture generation scheduling
    '''

    def _generation_loop(self):
        """
        Thread 2: Monitor available audio and trigger gesture generation windows.
        
        Generation strategy:
        - Window indices are initialized automatically when utterance is created
        - Wait until enough audio samples are available to fill next_window_end_sample
        - Snapshot the required audio samples, release lock, and generate gestures
        - After generation, acquire lock and write waypoints back (if utterance still exists)
        - Update window indices using utterance.update_window_indices() for next generation
        
        Desired behaviors:
        - start_margin controls where the first window begins (not from t=0)
        - If utterance tail is insufficient for a full window, no generation is triggered
        - Generation naturally stays ahead of playback since it's faster than realtime
        """
        while self.running:
            time.sleep(0.05)  # 50ms tick
            
            # Step 1: Check if we have enough samples for the next window
            should_generate = False
            utterance_id = None
            audio_snapshot = None
            window_start_sample = None
            window_start_frame = None
            
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
                
                # Calculate starting frame index for this window
                # Frames are indexed from the beginning of the utterance at gesture_fps
                window_start_frame = int(window_start_sample / utterance.sample_rate * utterance.gesture_fps)
                
                # Snapshot the audio samples we need
                audio_snapshot = utterance.get_audio_window(window_start_sample, window_end_sample)
            
            # Step 2: Generate gestures without holding the lock
            if should_generate and len(audio_snapshot) > 0:
                # Generate gestures for this window (releases lock during generation)
                waypoints = self._generate_gesture_window_from_audio(
                    audio_snapshot, window_start_frame, utterance_id
                )
                
                # Step 3: Write waypoints back and update generation state
                with self.utterance_lock:
                    utterance = self.current_utterance
                    
                    # Check if this utterance is still current (not cancelled)
                    if utterance is None or utterance.utterance_id != utterance_id:
                        # Utterance was cancelled, discard waypoints
                        continue
                    
                    # Write waypoints
                    if waypoints and utterance.gesture_waypoints is not None:
                        utterance.gesture_waypoints.add_waypoints(waypoints)
                    
                    # Update window indices for next generation
                    utterance.update_window_indices()

    
    def _generate_gesture_window_from_audio(
        self, 
        audio_snapshot: np.ndarray, 
        start_frame: int,
        utterance_id: int
    ) -> List[GestureWaypoint]:
        """
        Generate gestures for a window of audio from a snapshot of audio samples.
        
        This method does NOT hold the utterance lock during generation.
        It works with a snapshot of audio samples taken from the utterance.
        
        Args:
            audio_snapshot: Audio samples for this window (numpy array, float32 in range [-1, 1])
            start_frame: Starting frame index for this window (in gesture_fps time)
            utterance_id: ID of the utterance (for fetching overlap frames)
        
        Returns:
            List of generated waypoints (only the non-overlapping window_step frames)
        """
        if len(audio_snapshot) == 0:
            return []
        
        # Resample and extract mel spectrogram (following DiffSHEG pipeline)
        aud = librosa.resample(audio_snapshot, orig_sr=self.audio_sr, target_sr=18000)
        mel = librosa.feature.melspectrogram(y=aud, sr=18000, hop_length=1200, n_mels=128)
        mel = mel[..., :-1]
        audio_emb = torch.from_numpy(np.swapaxes(mel, -1, -2))
        audio_emb = audio_emb.unsqueeze(0).to(self.device)
        
        # Audio embedding should match window size
        audio_window = audio_emb
        
        # Pad or trim to window size
        if audio_window.shape[1] < self.window_size:
            padding = self.window_size - audio_window.shape[1]
            audio_window = torch.cat([
                audio_window,
                torch.zeros(1, padding, audio_window.shape[2], device=self.device)
            ], dim=1)
        elif audio_window.shape[1] > self.window_size:
            audio_window = audio_window[:, :self.window_size, :]
        
        # Prepare inputs for DiffSHEG
        B, T, _ = audio_window.shape
        C = self.opt.net_dim_pose
        motions = torch.zeros((B, T, C), device=self.device)
        
        # Person ID (default to 0, can be configured)
        p_id = torch.zeros((1, 1), device=self.device)
        p_id = self.model.one_hot(p_id, self.opt.speaker_dim).to(self.device)
        
        # Additional conditioning (e.g., HuBERT features)
        add_cond = {}
        
        # Inpainting for overlap - use last overlap_len frames from previous window
        inpaint_dict = {}
        if self.overlap_len > 0 and start_frame > 0:
            # Get previous waypoints for smooth transition
            # Need to acquire lock briefly to read waypoints
            with self.utterance_lock:
                if self.current_utterance and self.current_utterance.utterance_id == utterance_id:
                    if self.current_utterance.gesture_waypoints is not None:
                        # We need the last overlap_len waypoints before start_frame
                        prev_waypoints = []
                        for i in range(self.overlap_len):
                            prev_frame = start_frame - self.overlap_len + i
                            if prev_frame >= 0:
                                # Find waypoint at this frame index
                                with self.current_utterance.gesture_waypoints.lock:
                                    for wp in self.current_utterance.gesture_waypoints.waypoints:
                                        if wp.waypoint_index == prev_frame:
                                            prev_waypoints.append(wp.gesture_data)
                                            break
                        
                        if len(prev_waypoints) == self.overlap_len:
                            inpaint_dict['gt'] = torch.zeros_like(motions)
                            inpaint_dict['outpainting_mask'] = torch.zeros_like(motions, dtype=torch.bool, device=self.device)
                            inpaint_dict['outpainting_mask'][:, :self.overlap_len, :] = True
                            
                            # Use the overlap frames from previous window
                            prev_frames = torch.from_numpy(np.stack(prev_waypoints)).to(self.device)
                            inpaint_dict['gt'][:, :self.overlap_len, :] = prev_frames.unsqueeze(0)
        
        # Generate gestures (no lock held during model inference)
        with torch.no_grad():
            outputs = self.model.generate_batch(
                audio_window, p_id, C, add_cond, inpaint_dict
            )
        
        outputs_np = outputs.cpu().numpy()[0]  # Shape: (window_size, C)
        
        # Create waypoints from the generated gestures
        # Only create waypoints for the non-overlapping part (window_step frames)
        waypoints = []
        for i in range(self.window_step):
            frame_index = start_frame + i
            timestamp = frame_index / self.gesture_fps  # Time in seconds from utterance start
            
            waypoint = GestureWaypoint(
                waypoint_index=frame_index,
                timestamp=timestamp,
                gesture_data=outputs_np[i]  # Shape: (C,)
            )
            waypoints.append(waypoint)
        
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
    # # Audio data should be numpy arrays (float32 in range [-1, 1] or int16)
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
