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


@dataclass
class AudioChunk:
    """Represents a single audio chunk from the dialogue system."""
    utterance_id: int
    chunk_index: int
    audio_data: np.ndarray  # Raw audio at original sample rate
    timestamp: float  # When this chunk was received
    duration: float  # Duration in seconds


@dataclass
class Utterance:
    """Tracks an ongoing or completed utterance."""
    utterance_id: int
    chunks: List[AudioChunk] = field(default_factory=list)
    is_cancelled: bool = False
    start_time: Optional[float] = None
    current_chunk_playing: int = 0
    generation_started: bool = False  # Whether we've started generating for this utterance
    start_margin: float = 0.5  # When to start generation relative to playback
    last_chunk_received_time: float = field(default_factory=time.time)  # Track when last chunk arrived
    gesture_waypoints: Optional['GestureWaypoints'] = None  # Gesture waypoints for this utterance
    
    def get_full_audio(self, sample_rate: int = 16000) -> np.ndarray:
        """Concatenate all audio chunks into a single array."""
        if not self.chunks:
            return np.array([])
        return np.concatenate([chunk.audio_data for chunk in self.chunks])
    
    def get_audio_up_to_chunk(self, chunk_index: int, sample_rate: int = 16000) -> np.ndarray:
        """Get audio data up to (and including) a specific chunk."""
        if chunk_index >= len(self.chunks):
            return self.get_full_audio(sample_rate)
        return np.concatenate([self.chunks[i].audio_data for i in range(chunk_index + 1)])


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
    
    def get_total_waypoints(self) -> int:
        """Return the total number of waypoints generated so far."""
        with self.lock:
            return len(self.waypoints)


class DiffSHEGRealtimeWrapper:
    """
    Wrapper for integrating DiffSHEG with real-time dialogue systems.
    
    Features:
    - Tracks utterance lifecycle and audio chunks
    - Schedules gesture generation with configurable start margin
    - Manages two threads: playback monitoring and generation scheduling
    """
    
    def __init__(
        self,
        diffsheg_model,
        opt,
        default_start_margin: float = 0.5,  # Default time ahead of playback to start generation
        audio_sr: int = 16000,
        device: str = "cuda",
        cleanup_timeout: float = 2.0  # Seconds after playback ends to auto-cleanup
    ):
        """
        Initialize the wrapper.
        
        Args:
            diffsheg_model: The DiffSHEG trainer instance (DDPMTrainer_beat)
            opt: Configuration options
            default_start_margin: Default start margin for new utterances. Used once per utterance
                                 to determine when to start generating gestures relative to playback.
                                 E.g., if playback starts at t=0, start generation when we have audio
                                 up to t=start_margin. Since generation is faster than realtime, this
                                 one-time delay ensures gestures are ready before playback needs them.
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
        
        # Generation state
        self.last_generated_waypoint_index: int = -1  # Track which waypoint index was last generated
        
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
        
        This clears the current utterance, cancellation history, and generation state.
        Useful for starting fresh or cleaning up after a session ends.
        """
        with self.utterance_lock:
            self.current_utterance = None
            self.cancelled_utterances.clear()
            self.last_generated_waypoint_index = -1
    
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
            start_margin: Optional start margin for this specific utterance. If None, uses default.
        """
        # Reject chunks for cancelled/timed-out utterances
        if utterance_id in self.cancelled_utterances:
            # Silently ignore - this is expected for late-arriving chunks after cancellation
            return
    
        chunk = AudioChunk(
            utterance_id=utterance_id,
            chunk_index=chunk_index,
            audio_data=audio_data,
            timestamp=time.time(),
            duration=duration
        )
        
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
                    gesture_waypoints=GestureWaypoints(gesture_fps=self.gesture_fps)
                )
                self.last_generated_waypoint_index = -1
            
            utterance = self.current_utterance
            
            # Update last chunk received time
            utterance.last_chunk_received_time = chunk.timestamp
            
            # Automatically start playback when first chunk arrives, we log the timestamp here
            if chunk_index == 0 and utterance.start_time is None:
                utterance.start_time = chunk.timestamp
            
            # Insert chunk at correct position
            while len(utterance.chunks) <= chunk_index:
                utterance.chunks.append(None)
            utterance.chunks[chunk_index] = chunk
    
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
                self.last_generated_waypoint_index = -1
    
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
                total_audio_duration = sum(chunk.duration for chunk in utterance.chunks if chunk is not None)
                

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
    
    def _generation_loop(self):
        """
        Thread 2: Monitor available audio and schedule gesture generation.
        Uses sliding window approach to generate gestures incrementally.
        
        Generation strategy:
        - Use start_margin once per utterance to determine when to start generation
        - Wait until we have audio_duration >= start_margin before starting generation
        - Once started, generate continuously since generation is faster than realtime
        - Generates waypoints at 15 FPS (one waypoint every ~66.67ms)
        - Uses sliding window with overlap for smooth transitions
        """
        while self.running:
            time.sleep(0.05)  # 50ms tick
            
            with self.utterance_lock:
                utterance = self.current_utterance
                
                if utterance is None:
                    continue
                
                # Check if we have any audio chunks
                if not utterance.chunks or all(c is None for c in utterance.chunks):
                    continue
                
                # Calculate total audio duration available
                audio_duration = sum(chunk.duration for chunk in utterance.chunks if chunk is not None)
                
                # Check if we should start generation for this utterance
                if not utterance.generation_started:
                    # Wait until we have enough audio (start_margin) before starting generation
                    if audio_duration < utterance.start_margin:
                        continue
                    # Mark generation as started
                    utterance.generation_started = True
                
                # Calculate the next waypoint index to generate
                # Each window generates window_step waypoints (e.g., 30 waypoints per window)
                next_waypoint_index = self.last_generated_waypoint_index + 1
                
                # Calculate which window this waypoint belongs to
                # For the first waypoint (index 0), we start at window 0
                # For waypoint 30, we need window 1, etc.
                if next_waypoint_index == 0:
                    window_start_frame = 0
                else:
                    # Each window produces window_step new waypoints
                    window_number = (next_waypoint_index + self.overlap_len) // self.window_step
                    window_start_frame = window_number * self.window_step
                
                # Check if we need to generate a new window
                # We need audio for the full window (window_size frames at gesture_fps)
                window_end_frame = window_start_frame + self.window_size
                window_end_time = window_end_frame / self.gesture_fps
                
                if window_end_time > audio_duration:
                    # Not enough audio for full window yet, wait for more chunks. This will elicit the desire behavior that, for the very tail of an utterance which does not contain a full window, we will simply ignore it.
                    continue
                
                # Generate this window (will produce waypoints from window_start_frame to window_start_frame + window_step)
                self._generate_gesture_window(window_start_frame)
                
                # Update last generated waypoint index
                # Each window generates window_step new waypoints
                self.last_generated_waypoint_index = window_start_frame + self.window_step - 1
    
    def _generate_gesture_window(self, start_frame: int):
        """
        Generate gestures for a single window of audio.
        
        A window generates window_size frames (e.g., 34 frames), but only the first
        window_step frames (e.g., 30 frames) are kept as new waypoints. The overlap_len
        frames (e.g., 4 frames) are used for smooth transitions but not stored as new waypoints.
        
        Args:
            start_frame: Starting frame index for this window (in gesture_fps time, e.g., 15 FPS)
        """
        with self.utterance_lock:
            utterance = self.current_utterance
            if utterance is None:
                return
            utterance_id = utterance.utterance_id
        
        # Calculate time range for this window
        # Frames are at gesture_fps (15 FPS), so frame 0 = 0s, frame 15 = 1s, etc.
        start_time = start_frame / self.gesture_fps
        window_end_frame = start_frame + self.window_size
        end_time = window_end_frame / self.gesture_fps
        
        # Get only the audio needed for this window
        start_sample = int(start_time * self.audio_sr)
        end_sample = int(end_time * self.audio_sr)
        
        audio_data = utterance.get_full_audio(self.audio_sr)
        if len(audio_data) == 0:
            return
        
        # Extract only the window we need
        audio_window_data = audio_data[start_sample:min(end_sample, len(audio_data))]
        
        # Resample and extract mel spectrogram (following DiffSHEG pipeline)
        aud = librosa.resample(audio_window_data, orig_sr=self.audio_sr, target_sr=18000)
        mel = librosa.feature.melspectrogram(y=aud, sr=18000, hop_length=1200, n_mels=128)
        mel = mel[..., :-1]
        audio_emb = torch.from_numpy(np.swapaxes(mel, -1, -2))
        audio_emb = audio_emb.unsqueeze(0).to(self.device)
        
        # Audio embedding should match window size
        audio_window = audio_emb
        
        # Pad if necessary
        if audio_window.shape[1] < self.window_size:
            padding = self.window_size - audio_window.shape[1]
            audio_window = torch.cat([
                audio_window,
                torch.zeros(1, padding, audio_window.shape[2], device=self.device)
            ], dim=1)
        
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
            with self.utterance_lock:
                if utterance.gesture_waypoints is not None:
                    # We need the last overlap_len waypoints before start_frame
                    prev_waypoints = []
                    for i in range(self.overlap_len):
                        prev_frame = start_frame - self.overlap_len + i
                        if prev_frame >= 0:
                            # Find waypoint at this frame index
                            with utterance.gesture_waypoints.lock:
                                for wp in utterance.gesture_waypoints.waypoints:
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
        
        # Generate gestures
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
        
        # Add waypoints to the utterance's gesture waypoints
        with self.utterance_lock:
            if utterance.gesture_waypoints is not None:
                utterance.gesture_waypoints.add_waypoints(waypoints)
    
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
            self.last_generated_waypoint_index = -1


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
