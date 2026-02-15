#!/usr/bin/env python3
"""
ui_components.py

Implements the graphical user interface controls for the Acoustic Imager.

This module defines:
- The Button and ButtonState classes
- The bottom control buttons (Camera, Source, Debug)
- The top-right MENU and its sub-controls (FPS, Gain, Screenshot, Record, Pause)
- Button layout, hover/active state handling, and drawing routines
- Mouse click handling for all UI interactions

It is responsible for:
- Updating UI state in response to user input
- Triggering actions such as toggling camera/source/debug, changing FPS/gain,
  taking screenshots, and starting/stopping/pausing recordings
- Rendering all UI elements onto the main output frame
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

import cv2
import numpy as np

from .video_recorder import VideoRecorder

# -------------------------------------------------------------
# Import config/state
# (Fallbacks included)
# -------------------------------------------------------------
try:
    from ..config import (
        WIDTH,
        HEIGHT,
        DB_BAR_WIDTH,
        BUTTON_HITPAD_PX,
        USE_CAMERA,
    )
except Exception:
    WIDTH = 1024
    HEIGHT = 600
    DB_BAR_WIDTH = 50
    BUTTON_HITPAD_PX = 14
    USE_CAMERA = True

try:
    from ..state import button_state
except Exception:
    button_state = None  # will be created below if missing


# ===============================================================
# 7. Screenshot helper
# ===============================================================
def save_screenshot(frame: np.ndarray, output_dir: Path) -> Optional[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"screenshot_{timestamp}.png"
    if cv2.imwrite(str(filepath), frame):
        print(f"Screenshot saved: {filepath}")
        return str(filepath)
    print(f"Failed to save screenshot: {filepath}")
    return None


# ===============================================================
# 8. UI Buttons + MENU dropdown
# ===============================================================
class ButtonState:
    def __init__(self):
        self.is_recording = False
        self.is_paused = False
        self.camera_enabled = USE_CAMERA
        self.source_mode = "SIM"  # SIM or SPI

        # MENU states
        self.menu_open = False
        self.fps_mode = "60"   # "30" | "60" | "MAX"
        self.gain_mode = "LOW"  # placeholder toggle
        self.debug_enabled = True


# If state.py didn't provide button_state, create it here
if button_state is None:
    button_state = ButtonState()


# Global UI registries (matches original)
buttons: Dict[str, "Button"] = {}
menu_buttons: Dict[str, "Button"] = {}

# FPS mapping (matches original)
FPS_MODE_TO_TARGET = {"30": 30, "60": 60}


def _rounded_rect(img, x, y, w, h, r, color, thickness=-1):
    # original implementation is just a rectangle (kept identical)
    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        color,
        thickness,
        lineType=cv2.LINE_AA
    )


class Button:
    def __init__(self, x, y, w, h, text):
        self.x, self.y, self.w, self.h = int(x), int(y), int(w), int(h)
        self.text = text
        self.is_hovered = False
        self.is_active = False

    def contains(self, mx, my) -> bool:
        pad = BUTTON_HITPAD_PX
        return (
            (self.x - pad) <= mx <= (self.x + self.w + pad) and
            (self.y - pad) <= my <= (self.y + self.h + pad)
        )

    def draw(self, frame: np.ndarray) -> None:
        base = (60, 60, 60)
        hover = (85, 85, 85)
        active = (70, 90, 70)
        border = (230, 230, 230)

        color = active if self.is_active else (hover if self.is_hovered else base)

        # Solid filled rounded rectangle (no overlay, no blending)
        _rounded_rect(frame, self.x, self.y, self.w, self.h,
                      r=10, color=color, thickness=-1)

        # Border
        _rounded_rect(frame, self.x, self.y, self.w, self.h,
                      r=10, color=border, thickness=2)

        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.52
        thick = 1
        tw, th = cv2.getTextSize(self.text, font, scale, thick)[0]
        tx = self.x + (self.w - tw) // 2
        ty = self.y + (self.h + th) // 2

        cv2.putText(frame, self.text, (tx, ty),
                    font, scale, (255, 255, 255),
                    thick, cv2.LINE_AA)


def init_buttons(left_width: int, camera_available: bool) -> None:
    buttons.clear()

    n = 3
    margin = 10
    y = HEIGHT - 50 - 10
    h = 50

    left_pad = DB_BAR_WIDTH + 12
    right_pad = 12
    avail = left_width - left_pad - right_pad

    w = (avail - (n - 1) * margin) // n
    w = int(max(160, min(240, w)))

    total = n * w + (n - 1) * margin
    x0 = left_pad + (avail - total) // 2

    cam_text = "Camera: N/A" if not camera_available else (
        "Camera: ON" if button_state.camera_enabled else "Camera: OFF"
    )
    buttons["camera"] = Button(x0 + 0 * (w + margin + 20), y, w, h, cam_text)
    buttons["camera"].is_active = camera_available and button_state.camera_enabled

    buttons["source"] = Button(x0 + 1 * (w + margin + 5), y, w, h, f"Source: {button_state.source_mode}")
    buttons["source"].is_active = True

    dbg_text = "DEBUG: ON" if button_state.debug_enabled else "DEBUG: OFF"
    buttons["debug"] = Button(x0 + 2 * (w + margin + 5), y, w, h, dbg_text)
    buttons["debug"].is_active = button_state.debug_enabled


def init_menu_buttons(left_width: int) -> None:
    menu_buttons.clear()

    menu_x = 590
    menu_y = 10
    menu_w = 220
    menu_h = 60

    menu_buttons["menu"] = Button(menu_x, menu_y, menu_w, menu_h, "MENU")

    item_h = 40
    gap = 8
    y0 = menu_y + menu_h + gap

    # Segmented FPS buttons (30 | 60 | MAX)
    seg_gap = 6
    seg_w = (menu_w - 2 * seg_gap) // 3
    menu_buttons["fps30"] = Button(menu_x + 0 * (seg_w + seg_gap), y0, seg_w, item_h, "30FPS")
    menu_buttons["fps60"] = Button(menu_x + 1 * (seg_w + seg_gap), y0, seg_w, item_h, "60FPS")
    menu_buttons["fpsmax"] = Button(menu_x + 2 * (seg_w + seg_gap), y0, seg_w, item_h, "MAX")

    # Gain toggle (full width)
    gain_y = y0 + (item_h + gap)
    menu_buttons["gain"] = Button(menu_x, gain_y, menu_w, item_h, f"GAIN: {button_state.gain_mode}")

    # Segmented tools under GAIN (SHOT | REC | PAUSE)
    tools_y = gain_y + (item_h + gap)
    tool_gap = 6
    tool_w = (menu_w - 2 * tool_gap) // 3

    menu_buttons["shot"] = Button(menu_x + 0 * (tool_w + tool_gap), tools_y, tool_w, item_h, "SHOT")
    menu_buttons["rec"] = Button(menu_x + 1 * (tool_w + tool_gap), tools_y, tool_w, item_h, "REC")
    menu_buttons["pause"] = Button(menu_x + 2 * (tool_w + tool_gap), tools_y, tool_w, item_h, "PAUSE")


def update_button_states(mx: int, my: int) -> None:
    for b in buttons.values():
        b.is_hovered = b.contains(mx, my)

    if "menu" in menu_buttons:
        menu_buttons["menu"].is_hovered = menu_buttons["menu"].contains(mx, my)

    keys = ("fps30", "fps60", "fpsmax", "gain", "shot", "rec", "pause")

    if button_state.menu_open:
        for k in keys:
            if k in menu_buttons:
                menu_buttons[k].is_hovered = menu_buttons[k].contains(mx, my)
    else:
        for k in keys:
            if k in menu_buttons:
                menu_buttons[k].is_hovered = False


def draw_buttons(frame: np.ndarray) -> None:
    for b in buttons.values():
        b.draw(frame)


def draw_menu(frame: np.ndarray) -> None:
    if "menu" not in menu_buttons:
        return

    menu_buttons["menu"].is_active = button_state.menu_open
    menu_buttons["menu"].draw(frame)

    if not button_state.menu_open:
        return

    x = menu_buttons["menu"].x
    y = menu_buttons["menu"].y + menu_buttons["menu"].h + 6
    w = menu_buttons["menu"].w
    h = 3 * 40 + 2 * 8 + 20

    # original ROI overlay behavior
    x0, y0 = x - 2, y - 2
    x1, y1 = x0 + (w + 4), y0 + (h + 4)

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(frame.shape[1], x1)
    y1 = min(frame.shape[0], y1)

    roi = frame[y0:y1, x0:x1]
    overlay = roi.copy()
    _rounded_rect(overlay, 0, 0, overlay.shape[1], overlay.shape[0], r=12, color=(20, 20, 20), thickness=-1)
    cv2.addWeighted(overlay, 0.55, roi, 0.45, 0, roi)

    menu_buttons["fps30"].is_active = (button_state.fps_mode == "30")
    menu_buttons["fps60"].is_active = (button_state.fps_mode == "60")
    menu_buttons["fpsmax"].is_active = (button_state.fps_mode == "MAX")

    menu_buttons["gain"].is_active = (button_state.gain_mode == "HIGH")

    # Tool button actives reflect current state
    menu_buttons["rec"].is_active = button_state.is_recording
    menu_buttons["pause"].is_active = button_state.is_paused

    menu_buttons["fps30"].draw(frame)
    menu_buttons["fps60"].draw(frame)
    menu_buttons["fpsmax"].draw(frame)
    menu_buttons["gain"].draw(frame)

    menu_buttons["shot"].draw(frame)
    menu_buttons["rec"].draw(frame)
    menu_buttons["pause"].draw(frame)


def draw_recording_timestamp(frame: np.ndarray, video_recorder: Optional[VideoRecorder]) -> None:
    """
    Draw recording timestamp below the menu when recording is active.
    Shows elapsed time in MM:SS format, with red dot indicator.
    """
    if video_recorder is None or not video_recorder.is_recording:
        return
    
    if "menu" not in menu_buttons:
        return
    
    # Get elapsed time
    elapsed = video_recorder.get_elapsed_time()
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    # Position below the last menu button row (SHOT/REC/PAUSE)
    menu_x = menu_buttons["menu"].x
    menu_w = menu_buttons["menu"].w
    
    # Find the bottom of the tools row
    tools_bottom = menu_buttons["rec"].y + menu_buttons["rec"].h + 10
    
    # Timestamp box dimensions
    timestamp_h = 35
    timestamp_y = tools_bottom
    
    # Draw semi-transparent background
    overlay = frame[timestamp_y:timestamp_y+timestamp_h, menu_x:menu_x+menu_w].copy()
    bg_color = (30, 30, 30) if not video_recorder.is_paused else (40, 40, 60)
    cv2.rectangle(overlay, (0, 0), (menu_w, timestamp_h), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, frame[timestamp_y:timestamp_y+timestamp_h, menu_x:menu_x+menu_w], 0.3, 0,
                    frame[timestamp_y:timestamp_y+timestamp_h, menu_x:menu_x+menu_w])
    
    # Draw border
    border_color = (200, 200, 200) if not video_recorder.is_paused else (100, 100, 200)
    cv2.rectangle(frame, (menu_x, timestamp_y), (menu_x + menu_w, timestamp_y + timestamp_h),
                  border_color, 1, cv2.LINE_AA)
    
    # Draw recording indicator (red dot or paused icon)
    indicator_x = menu_x + 10
    indicator_y = timestamp_y + timestamp_h // 2
    
    if video_recorder.is_paused:
        # Draw pause icon (two vertical bars)
        bar_w = 3
        bar_h = 10
        bar_gap = 4
        cv2.rectangle(frame, 
                     (indicator_x, indicator_y - bar_h//2),
                     (indicator_x + bar_w, indicator_y + bar_h//2),
                     (100, 150, 255), -1, cv2.LINE_AA)
        cv2.rectangle(frame,
                     (indicator_x + bar_w + bar_gap, indicator_y - bar_h//2),
                     (indicator_x + 2*bar_w + bar_gap, indicator_y + bar_h//2),
                     (100, 150, 255), -1, cv2.LINE_AA)
    else:
        # Draw pulsing red dot (recording)
        import time
        pulse = int((time.time() * 2) % 2)  # Blink every 0.5 seconds
        if pulse:
            cv2.circle(frame, (indicator_x + 5, indicator_y), 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (indicator_x + 5, indicator_y), 6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw timestamp text
    timestamp_text = f"{minutes:02d}:{seconds:02d}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thick = 2
    text_color = (255, 255, 255) if not video_recorder.is_paused else (150, 150, 255)
    
    # Center the text
    (text_w, text_h), _ = cv2.getTextSize(timestamp_text, font, font_scale, font_thick)
    text_x = menu_x + (menu_w - text_w) // 2 + 10  # Offset for indicator
    text_y = timestamp_y + (timestamp_h + text_h) // 2
    
    cv2.putText(frame, timestamp_text, (text_x, text_y),
                font, font_scale, text_color, font_thick, cv2.LINE_AA)


# ===============================================================
# Click handlers
# ===============================================================
def handle_menu_click(
    x: int,
    y: int,
    current_frame: Optional[np.ndarray],
    output_dir: Optional[Path],
    video_recorder: Optional[VideoRecorder],
    width: int,
    height: int,
) -> Optional[VideoRecorder]:
    """
    Returns (possibly newly created) video_recorder, same logic as original.
    """
    if "menu" not in menu_buttons:
        return video_recorder

    if menu_buttons["menu"].contains(x, y):
        button_state.menu_open = not button_state.menu_open
        return video_recorder

    if not button_state.menu_open:
        return video_recorder

    # FPS segmented
    if "fps30" in menu_buttons and menu_buttons["fps30"].contains(x, y):
        button_state.fps_mode = "30"
        return video_recorder
    if "fps60" in menu_buttons and menu_buttons["fps60"].contains(x, y):
        button_state.fps_mode = "60"
        return video_recorder
    if "fpsmax" in menu_buttons and menu_buttons["fpsmax"].contains(x, y):
        button_state.fps_mode = "MAX"
        return video_recorder

    # Gain toggle
    if "gain" in menu_buttons and menu_buttons["gain"].contains(x, y):
        button_state.gain_mode = "HIGH" if button_state.gain_mode == "LOW" else "LOW"
        menu_buttons["gain"].text = f"GAIN: {button_state.gain_mode}"
        return video_recorder

    # SHOT
    if "shot" in menu_buttons and menu_buttons["shot"].contains(x, y):
        if current_frame is not None and output_dir is not None:
            save_screenshot(current_frame, output_dir)
        return video_recorder

    # REC
    if "rec" in menu_buttons and menu_buttons["rec"].contains(x, y):
        if video_recorder is None and output_dir is not None:
            video_recorder = VideoRecorder(output_dir, width, height, fps=30)

        if video_recorder is None:
            return None

        button_state.is_recording = not button_state.is_recording
        if button_state.is_recording:
            if video_recorder.start_recording():
                button_state.is_paused = False
            else:
                button_state.is_recording = False
        else:
            video_recorder.stop_recording()
            button_state.is_paused = False
        return video_recorder

    # PAUSE
    if "pause" in menu_buttons and menu_buttons["pause"].contains(x, y):
        if not button_state.is_recording or video_recorder is None:
            return video_recorder

        button_state.is_paused = not button_state.is_paused
        if button_state.is_paused:
            video_recorder.pause_recording()
            menu_buttons["pause"].text = "RESUME"
        else:
            video_recorder.resume_recording()
            menu_buttons["pause"].text = "PAUSE"
        return video_recorder

    return video_recorder


def handle_button_click(
    x: int,
    y: int,
    current_frame: Optional[np.ndarray],
    output_dir: Optional[Path],
    camera_available: bool,
    video_recorder: Optional[VideoRecorder],
    width: int,
    height: int,
    on_source_to_spi=None,
    on_source_to_sim=None,
) -> Optional[VideoRecorder]:
    """
    Handles button clicks:
      - Calls handle_menu_click first
      - Camera toggle
      - Source toggle (SIM <-> SPI) and calls optional callbacks
      - Debug toggle

    Returns updated video_recorder (may be created by menu click).
    """
    video_recorder = handle_menu_click(
        x, y,
        current_frame=current_frame,
        output_dir=output_dir,
        video_recorder=video_recorder,
        width=width,
        height=height,
    )

    if "camera" in buttons and buttons["camera"].contains(x, y):
        if camera_available:
            button_state.camera_enabled = not button_state.camera_enabled
            buttons["camera"].is_active = button_state.camera_enabled
            buttons["camera"].text = "Camera: ON" if button_state.camera_enabled else "Camera: OFF"
        return video_recorder

    if "source" in buttons and buttons["source"].contains(x, y):
        button_state.source_mode = "SPI" if button_state.source_mode == "SIM" else "SIM"
        buttons["source"].text = f"Source: {button_state.source_mode}"

        if button_state.source_mode == "SPI":
            if callable(on_source_to_spi):
                on_source_to_spi()
        else:
            if callable(on_source_to_sim):
                on_source_to_sim()
        return video_recorder

    if "debug" in buttons and buttons["debug"].contains(x, y):
        button_state.debug_enabled = not button_state.debug_enabled
        buttons["debug"].is_active = button_state.debug_enabled
        buttons["debug"].text = "DEBUG: ON" if button_state.debug_enabled else "DEBUG: OFF"
        return video_recorder

    return video_recorder
