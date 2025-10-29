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
class GestureSegment:
    """Represents generated gestures for a time window."""
    utterance_id: int
    start_frame: int  # Frame index in the full utterance
    end_frame: int
    gestures: np.ndarray  # Generated gesture data (T, C)
    audio_chunk_range: Tuple[int, int]  # (start_chunk, end_chunk) indices


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
        fps: int = 30,  # Gesture playback FPS
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
            fps: Target FPS for gesture playback
            audio_sr: Audio sample rate
            device: Computing device
            cleanup_timeout: Seconds to wait after playback ends before auto-cleanup
        """
        self.model = diffsheg_model
        self.opt = opt
        self.default_start_margin = default_start_margin
        self.fps = fps
        self.audio_sr = audio_sr
        self.device = device
        self.cleanup_timeout = cleanup_timeout
        
        # Current utterance tracking (only keep the latest)
        self.current_utterance: Optional[Utterance] = None
        self.current_gestures: List[GestureSegment] = []
        self.utterance_lock = threading.Lock()
        
        # Generation state
        self.last_generated_time: float = 0.0  # Track generation progress for current utterance
        
        # Threading
        self.running = False
        self.playback_thread: Optional[threading.Thread] = None
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
        self.playback_thread = threading.Thread(target=self._playback_monitor_loop, daemon=True)
        self.generation_thread = threading.Thread(target=self._generation_loop, daemon=True)
        self.playback_thread.start()
        self.generation_thread.start()
        
    def stop(self):
        """Stop the wrapper threads."""
        self.running = False
        if self.playback_thread:
            self.playback_thread.join(timeout=2.0)
        if self.generation_thread:
            self.generation_thread.join(timeout=2.0)
    
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
        chunk = AudioChunk(
            utterance_id=utterance_id,
            chunk_index=chunk_index,
            audio_data=audio_data,
            timestamp=time.time(),
            duration=duration
        )
        
        with self.utterance_lock:
            # If this is a new utterance, discard the old one
            if self.current_utterance is None or self.current_utterance.utterance_id != utterance_id:
                margin = start_margin if start_margin is not None else self.default_start_margin
                self.current_utterance = Utterance(
                    utterance_id=utterance_id,
                    start_margin=margin
                )
                self.current_gestures = []
                self.last_generated_time = 0.0
            
            utterance = self.current_utterance
            
            # Update last chunk received time
            utterance.last_chunk_received_time = time.time()
            
            # Automatically start playback when first chunk arrives
            if chunk_index == 0 and utterance.start_time is None:
                utterance.start_time = time.time()
            
            # Insert chunk at correct position
            while len(utterance.chunks) <= chunk_index:
                utterance.chunks.append(None)
            utterance.chunks[chunk_index] = chunk
    

    
    def cancel_utterance(self, utterance_id: int):
        """
        Cancel an utterance and discard its audio chunks.
        This is the only explicit signal for utterance termination.
        
        Args:
            utterance_id: The utterance to cancel (msg_idx in your system)
        """
        with self.utterance_lock:
            if self.current_utterance and self.current_utterance.utterance_id == utterance_id:
                self.current_utterance = None
                self.current_gestures = []
                self.last_generated_time = 0.0
    
    def _playback_monitor_loop(self):
        """
        Thread 1: Monitor playback state and wall time.
        Tracks which chunk is currently playing based on wall time.
        Playback starts automatically when first chunk (index 0) arrives.
        Automatically cleans up after playback ends.
        TODO: Schedule gesture playback (not implemented yet).
        """
        while self.running:
            time.sleep(0.01)  # 10ms tick
            
            should_cleanup = False
            
            with self.utterance_lock:
                utterance = self.current_utterance
                
                if utterance is None:
                    continue
                
                # Skip if playback hasn't started yet (no chunk 0 received)
                if utterance.start_time is None:
                    continue
                
                # Calculate current playback position
                elapsed_time = time.time() - utterance.start_time
                
                # Calculate total audio duration
                total_audio_duration = sum(chunk.duration for chunk in utterance.chunks if chunk is not None)
                
                # Update which chunk is playing based on elapsed time
                cumulative_duration = 0.0
                for i, chunk in enumerate(utterance.chunks):
                    if chunk is None:
                        continue
                    cumulative_duration += chunk.duration
                    if cumulative_duration >= elapsed_time:
                        utterance.current_chunk_playing = i
                        break
                
                # Auto-cleanup: if playback has passed all audio AND no new chunks for cleanup_timeout
                if elapsed_time > total_audio_duration:
                    time_since_last_chunk = time.time() - utterance.last_chunk_received_time
                    if time_since_last_chunk > self.cleanup_timeout:
                        should_cleanup = True
                
                # TODO: Schedule gesture playback based on elapsed_time
                # This would involve retrieving generated gestures and sending them
                # to the robot control system at the appropriate times
                # Example:
                # current_frame = int(elapsed_time * self.gesture_fps)
                # gesture = self._get_gesture_for_frame(current_frame)
                # if gesture is not None:
                #     self.send_to_robot(gesture)
            
            if should_cleanup:
                self._cleanup_current_utterance()
    
    def _generation_loop(self):
        """
        Thread 2: Monitor available audio and schedule gesture generation.
        Uses sliding window approach to generate gestures incrementally.
        
        Generation strategy:
        - Use start_margin once per utterance to determine when to start generation
        - Wait until we have audio_duration >= start_margin before starting generation
        - Once started, generate continuously since generation is faster than realtime
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
                
                # Get the last frame we generated
                last_generated_frame = int(self.last_generated_time * self.gesture_fps)
                
                # Check if we need to generate a new window
                next_window_start = last_generated_frame
                next_window_end = next_window_start + self.window_size
                
                # Check if we have enough audio for the full window
                window_end_time = next_window_end / self.gesture_fps
                if window_end_time > audio_duration:
                    # Not enough audio for full window yet, wait for more chunks
                    continue
                
                # Generate this window
                self._generate_gesture_window(next_window_start)
                self.last_generated_time = (next_window_start + self.window_step) / self.gesture_fps
    
    def _generate_gesture_window(self, start_frame: int):
        """
        Generate gestures for a single window of audio.
        
        Args:
            start_frame: Starting frame index for this window
        """
        with self.utterance_lock:
            utterance = self.current_utterance
            if utterance is None:
                return
            utterance_id = utterance.utterance_id
        
        # Calculate time range for this window
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
        
        # Inpainting for overlap
        inpaint_dict = {}
        if self.overlap_len > 0 and start_frame > 0:
            # Get previous window's last frames for smooth transition
            with self.utterance_lock:
                if len(self.current_gestures) > 0:
                    prev_segment = self.current_gestures[-1]
                    if prev_segment.gestures.shape[0] >= self.overlap_len:
                        inpaint_dict['gt'] = torch.zeros_like(motions)
                        inpaint_dict['outpainting_mask'] = torch.zeros_like(motions, dtype=torch.bool, device=self.device)
                        inpaint_dict['outpainting_mask'][:, :self.overlap_len, :] = True
                        
                        # Use last overlap_len frames from previous segment
                        prev_frames = torch.from_numpy(prev_segment.gestures[-self.overlap_len:]).to(self.device)
                        inpaint_dict['gt'][:, :self.overlap_len, :] = prev_frames.unsqueeze(0)
        
        # Generate gestures
        with torch.no_grad():
            outputs = self.model.generate_batch(
                audio_window, p_id, C, add_cond, inpaint_dict
            )
        
        outputs_np = outputs.cpu().numpy()
        
        # Determine which audio chunks this corresponds to
        start_chunk_idx = self._time_to_chunk_index(utterance, start_time)
        end_chunk_idx = self._time_to_chunk_index(utterance, end_time)
        
        # Create gesture segment
        segment = GestureSegment(
            utterance_id=utterance_id,
            start_frame=start_frame,
            end_frame=window_end_frame,
            gestures=outputs_np[0, :self.window_step, :],  # Only keep non-overlapping part
            audio_chunk_range=(start_chunk_idx, end_chunk_idx)
        )
        
        # Store generated gestures
        with self.utterance_lock:
            self.current_gestures.append(segment)
    
    def _time_to_chunk_index(self, utterance: Utterance, time_seconds: float) -> int:
        """Convert time in seconds to chunk index."""
        cumulative_duration = 0.0
        for i, chunk in enumerate(utterance.chunks):
            if chunk is None:
                continue
            cumulative_duration += chunk.duration
            if cumulative_duration >= time_seconds:
                return i
        return len(utterance.chunks) - 1
    
    def get_current_gestures(self) -> Optional[np.ndarray]:
        """
        Get all generated gestures for the current utterance concatenated.
        
        Returns:
            np.ndarray of shape (T, C) or None if no gestures available
        """
        with self.utterance_lock:
            if not self.current_gestures:
                return None
            return np.concatenate([seg.gestures for seg in self.current_gestures], axis=0)
    
    def _cleanup_current_utterance(self):
        """
        Internal cleanup method called automatically by playback monitor.
        Cleans up current utterance data after playback naturally ends.
        """
        with self.utterance_lock:
            self.current_utterance = None
            self.current_gestures = []
            self.last_generated_time = 0.0


# Example usage
if __name__ == "__main__":
    # This is a demonstration of how to use the wrapper
    # In practice, you would initialize this with your trained DiffSHEG model
    
    # Example: Initialize wrapper (pseudo-code)
    # wrapper = DiffSHEGRealtimeWrapper(
    #     diffsheg_model=your_trained_model,
    #     opt=your_opt_config,
    #     start_margin=0.5,
    #     fps=30,
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
