"""
Playback state management for gesture playback.

States:
- IDLE: No active utterance and no blending in progress. Neutral position maintained.
- IN_UTTERANCE: Currently playing back gestures for an active utterance.
- BLENDING_TO_NEUTRAL: Transitioning from last executed gesture to neutral position.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.EnumWithValEq import EnumWithValEq


class PlaybackState(EnumWithValEq):
    """
    Enum for tracking the current playback state of the gesture system.
    
    States:
    - IDLE: No active utterance, no blending. System is at neutral position.
    - IN_UTTERANCE: Currently executing waypoints for an active utterance.
    - BLENDING_TO_NEUTRAL: Transitioning from last gesture to neutral position.
    """
    IDLE = "idle"
    IN_UTTERANCE = "in_utterance"
    BLENDING_TO_NEUTRAL = "blending_to_neutral"
