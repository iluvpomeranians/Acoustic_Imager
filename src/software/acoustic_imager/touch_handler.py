#!/usr/bin/env python3
"""
touch_handler.py

Touch gesture recognition for touchscreen interactions.
Provides pinch-to-zoom, swipe, and multi-touch handling.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import time


@dataclass
class TouchPoint:
    """Represents a single touch point."""
    id: int
    x: float
    y: float
    timestamp: float


class TouchGestureHandler:
    """
    Handles multi-touch gestures like pinch-to-zoom and swipe.
    
    Usage:
        handler = TouchGestureHandler()
        
        # On touch down
        handler.add_touch(touch_id, x, y)
        
        # On touch move
        handler.update_touch(touch_id, x, y)
        gesture = handler.detect_gesture()
        
        # On touch up
        handler.remove_touch(touch_id)
    """
    
    def __init__(self, pinch_threshold: float = 20.0, swipe_threshold: float = 50.0):
        """
        Args:
            pinch_threshold: Minimum distance change (pixels) to detect pinch
            swipe_threshold: Minimum distance (pixels) to detect swipe
        """
        self.active_touches: Dict[int, TouchPoint] = {}
        self.initial_touches: Dict[int, TouchPoint] = {}
        self.pinch_threshold = pinch_threshold
        self.swipe_threshold = swipe_threshold
        
        # Gesture state
        self.last_pinch_distance: Optional[float] = None
        self.gesture_start_time: Optional[float] = None
        
    def add_touch(self, touch_id: int, x: float, y: float) -> None:
        """Register a new touch point."""
        touch = TouchPoint(touch_id, x, y, time.time())
        self.active_touches[touch_id] = touch
        self.initial_touches[touch_id] = TouchPoint(touch_id, x, y, time.time())
        
        if len(self.active_touches) == 2:
            # Two fingers down - potential pinch gesture
            self.gesture_start_time = time.time()
            self.last_pinch_distance = self._get_distance_between_touches()
    
    def update_touch(self, touch_id: int, x: float, y: float) -> None:
        """Update an existing touch point's position."""
        if touch_id in self.active_touches:
            self.active_touches[touch_id].x = x
            self.active_touches[touch_id].y = y
            self.active_touches[touch_id].timestamp = time.time()
    
    def remove_touch(self, touch_id: int) -> None:
        """Remove a touch point."""
        self.active_touches.pop(touch_id, None)
        self.initial_touches.pop(touch_id, None)
        
        if len(self.active_touches) < 2:
            self.last_pinch_distance = None
            self.gesture_start_time = None
    
    def clear(self) -> None:
        """Clear all touch points."""
        self.active_touches.clear()
        self.initial_touches.clear()
        self.last_pinch_distance = None
        self.gesture_start_time = None
    
    def _get_distance_between_touches(self) -> Optional[float]:
        """Calculate distance between two touch points."""
        if len(self.active_touches) != 2:
            return None
        
        touches = list(self.active_touches.values())
        dx = touches[1].x - touches[0].x
        dy = touches[1].y - touches[0].y
        return math.sqrt(dx * dx + dy * dy)
    
    def _get_midpoint(self) -> Optional[Tuple[float, float]]:
        """Get midpoint between two touch points."""
        if len(self.active_touches) != 2:
            return None
        
        touches = list(self.active_touches.values())
        mid_x = (touches[0].x + touches[1].x) / 2
        mid_y = (touches[0].y + touches[1].y) / 2
        return (mid_x, mid_y)
    
    def detect_gesture(self) -> Optional[Dict]:
        """
        Detect current gesture based on active touches.
        
        Returns:
            Dict with gesture info, or None if no gesture detected.
            
            Pinch gesture:
                {
                    'type': 'pinch',
                    'scale': float,  # >1.0 = zoom out, <1.0 = zoom in
                    'distance_delta': float,  # Change in distance between fingers
                    'midpoint': (x, y),  # Center point of pinch
                }
            
            Swipe gesture:
                {
                    'type': 'swipe',
                    'direction': str,  # 'up', 'down', 'left', 'right'
                    'distance': float,
                    'velocity': float,  # pixels per second
                }
        """
        # PINCH DETECTION (two fingers)
        if len(self.active_touches) == 2:
            current_distance = self._get_distance_between_touches()
            
            if current_distance is not None and self.last_pinch_distance is not None:
                distance_delta = current_distance - self.last_pinch_distance
                
                if abs(distance_delta) > self.pinch_threshold:
                    scale = current_distance / self.last_pinch_distance
                    midpoint = self._get_midpoint()
                    
                    # Update for next detection
                    self.last_pinch_distance = current_distance
                    
                    return {
                        'type': 'pinch',
                        'scale': scale,
                        'distance_delta': distance_delta,
                        'midpoint': midpoint,
                    }
        
        # SWIPE DETECTION (one finger)
        elif len(self.active_touches) == 1 and len(self.initial_touches) == 1:
            touch_id = list(self.active_touches.keys())[0]
            current = self.active_touches[touch_id]
            initial = self.initial_touches[touch_id]
            
            dx = current.x - initial.x
            dy = current.y - initial.y
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance > self.swipe_threshold:
                elapsed = current.timestamp - initial.timestamp
                velocity = distance / elapsed if elapsed > 0 else 0
                
                # Determine dominant direction
                if abs(dx) > abs(dy):
                    direction = 'right' if dx > 0 else 'left'
                else:
                    direction = 'down' if dy > 0 else 'up'
                
                return {
                    'type': 'swipe',
                    'direction': direction,
                    'distance': distance,
                    'velocity': velocity,
                    'delta_x': dx,
                    'delta_y': dy,
                }
        
        return None
    
    def get_touch_count(self) -> int:
        """Get number of active touches."""
        return len(self.active_touches)
    
    def is_pinching(self) -> bool:
        """Check if currently in pinch gesture."""
        return len(self.active_touches) == 2


# ===============================================================
# Simplified OpenCV-compatible touch simulation
# ===============================================================
class MouseTouchSimulator:
    """
    Simulates multi-touch using mouse for testing on non-touch systems.
    
    Usage:
        - Left click: Primary touch
        - Right click: Secondary touch (for pinch simulation)
    """
    
    def __init__(self):
        self.left_down = False
        self.right_down = False
        self.left_pos = (0, 0)
        self.right_pos = (0, 0)
    
    def get_simulated_touches(self) -> Dict[int, Tuple[int, int]]:
        """Get current simulated touch points as {touch_id: (x, y)}."""
        touches = {}
        if self.left_down:
            touches[0] = self.left_pos
        if self.right_down:
            touches[1] = self.right_pos
        return touches


# ===============================================================
# Utility functions
# ===============================================================
def is_point_in_rect(x: float, y: float, rect_x: int, rect_y: int, 
                     rect_w: int, rect_h: int) -> bool:
    """Check if point is inside rectangle."""
    return rect_x <= x <= rect_x + rect_w and rect_y <= y <= rect_y + rect_h


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(value, max_val))
