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
from typing import Optional, Dict, List, Tuple
import time

import cv2
import numpy as np

from .video_recorder import VideoRecorder

from ..config import SOURCE_MODES, SOURCE_DEFAULT

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

_GRAD_CACHE: dict[tuple[int, int, int, int, int], np.ndarray] = {}
# key = (w, h, b, g, r)
def _get_grad(w: int, h: int, color: tuple[int, int, int]) -> np.ndarray:
    key = (w, h, int(color[0]), int(color[1]), int(color[2]))
    g = _GRAD_CACHE.get(key)
    if g is not None:
        return g

    top = np.clip(np.array(color, np.float32) * 1.15, 0, 255)
    bot = np.clip(np.array(color, np.float32) * 0.85, 0, 255)

    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]  # (h,1,1)
    grad = (top[None, None, :] * (1.0 - t) + bot[None, None, :] * t).astype(np.uint8)  # (h,1,3)
    grad = np.repeat(grad, w, axis=1)  # (h,w,3)

    _GRAD_CACHE[key] = grad
    return grad

_REC_HUD_CACHE: dict[tuple[int, int, bool], np.ndarray] = {}
# key = (w, h, paused) -> hud background image

_REC_TEXT_SIZE_CACHE: dict[str, tuple[int, int]] = {}


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
# Global UI registries
buttons: Dict[str, "Button"] = {}
menu_buttons: Dict[str, "Button"] = {}

# FPS mapping
FPS_MODE_TO_TARGET = {"30": 30, "60": 60}


def _rounded_rect(img, x, y, w, h, r, color, thickness=-1):
    # original implementation is just a rectangle (kept identical)
    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        color,
        thickness,
        lineType=cv2.LINE_8
    )


def _draw_camera_icon(frame: np.ndarray, cx: int, cy: int, size: int = 12):
    """Draw a camera lens icon."""
    # Camera body (rectangle)
    cv2.rectangle(frame, (cx - size, cy - size//2), (cx + size, cy + size//2), (255, 255, 255), -1, cv2.LINE_AA)
    # Lens (circle)
    cv2.circle(frame, (cx, cy), size//2, (200, 200, 200), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), size//2, (0, 0, 0), 1, cv2.LINE_AA)
    # Flash indicator
    cv2.rectangle(frame, (cx - size, cy - size//2 - 3), (cx - size + 4, cy - size//2), (255, 255, 255), -1, cv2.LINE_AA)


def _draw_rec_icon(frame: np.ndarray, cx: int, cy: int, size: int = 8, is_active: bool = False):
    """Draw a red recording dot icon."""
    color = (0, 0, 255) if is_active else (255, 255, 255)
    cv2.circle(frame, (cx, cy), size, color, -1, cv2.LINE_AA)
    # Border
    border_color = (255, 255, 255) if is_active else (0, 0, 0)
    cv2.circle(frame, (cx, cy), size, border_color, 1, cv2.LINE_AA)


def _draw_pause_icon(frame: np.ndarray, cx: int, cy: int, size: int = 10):
    """Draw pause bars icon (||)."""
    bar_w = size // 3
    bar_h = size
    gap = size // 3
    
    # Left bar
    cv2.rectangle(frame, 
                 (cx - gap - bar_w, cy - bar_h//2),
                 (cx - gap, cy + bar_h//2),
                 (255, 255, 255), -1, cv2.LINE_AA)
    # Right bar
    cv2.rectangle(frame,
                 (cx + gap, cy - bar_h//2),
                 (cx + gap + bar_w, cy + bar_h//2),
                 (255, 255, 255), -1, cv2.LINE_AA)


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

    def draw(self, frame: np.ndarray, transparent: bool = False, active_color: Optional[tuple] = None, icon_type: Optional[str] = None) -> None:
        base = (60, 60, 60)
        hover = (85, 85, 85)
        active = active_color if active_color is not None else (40, 200, 60)
        border = (230, 230, 230)

        color = active if self.is_active else (hover if self.is_hovered else base)

        x, y, w, h = self.x, self.y, self.w, self.h

        if transparent:
            # Transparent style (like HUD pills)
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(frame.shape[1], x + w)
            y1 = min(frame.shape[0], y + h)
            
            if x1 > x0 and y1 > y0:
                roi = frame[y0:y1, x0:x1]
                overlay = np.empty_like(roi)
                overlay[:] = color
                # Use higher opacity for active buttons to make them more obvious
                alpha = 0.25 if self.is_active else 0.12
                cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)
        else:
            # Solid style with gradient (original)
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(frame.shape[1], x + w)
            y1 = min(frame.shape[0], y + h)
            if x1 > x0 and y1 > y0:
                roi = frame[y0:y1, x0:x1]
                roi[:] = _get_grad(roi.shape[1], roi.shape[0], color)

        # ---- border (keep cheap line type) ----
        # Use matching border color for active buttons
        if self.is_active:
            if active_color is not None:
                # Custom active color - make border brighter version
                border_color = tuple(min(255, int(c * 1.3)) for c in active_color)
            else:
                border_color = (80, 255, 100)  # Default bright green
        else:
            border_color = border
        _rounded_rect(frame, x, y, w, h, r=10, color=border_color, thickness=2)

        # ---- Draw icon if specified ----
        if icon_type:
            cx = x + w // 2
            cy = y + h // 2
            
            if icon_type == "camera":
                _draw_camera_icon(frame, cx, cy, size=10)
            elif icon_type == "rec":
                _draw_rec_icon(frame, cx, cy, size=7, is_active=self.is_active)
            elif icon_type == "pause":
                _draw_pause_icon(frame, cx, cy, size=9)
        else:
            # ---- text (AA is surprisingly expensive; use LINE_8) ----
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.52
            thick = 1
            tw, th = cv2.getTextSize(self.text, font, scale, thick)[0]
            tx = x + (w - tw) // 2
            ty = y + (h + th) // 2

            cv2.putText(frame, self.text, (tx, ty),
                        font, scale, (255, 255, 255),
                        thick, cv2.LINE_8)


def init_buttons(left_width: int, camera_available: bool) -> None:
    buttons.clear()
    return


def init_menu_buttons(left_width: int, frame_height: int = None) -> None:
    menu_buttons.clear()
    
    # Use actual frame height if provided, otherwise fall back to config HEIGHT
    actual_height = frame_height if frame_height is not None else HEIGHT
    
    menu_w = 180
    menu_h = 50
    menu_margin_x = 15
    menu_margin_bottom = 5  
    menu_x = left_width - menu_w - menu_margin_x
    menu_y = actual_height - menu_h - menu_margin_bottom

    menu_buttons["menu"] = Button(menu_x, menu_y, menu_w, menu_h, "MENU")

    item_h = 40
    gap = 8
    
    # Calculate total dropdown height to position it above the menu button
    total_items = 8  # FPS row + GAIN + COLORMAP + CAM + SOURCE + DEBUG + SHOT/REC/PAUSE row + GALLERY
    dropdown_h = total_items * (item_h + gap) + gap
    
    # Position dropdown above menu button
    dropdown_y = menu_y - dropdown_h - gap
    y0 = dropdown_y

    # Segmented FPS buttons (30 | 60 | MAX)
    seg_gap = 6
    seg_w = (menu_w - 2 * seg_gap) // 3
    menu_buttons["fps30"] = Button(menu_x + 0 * (seg_w + seg_gap), y0, seg_w, item_h, "30FPS")
    menu_buttons["fps60"] = Button(menu_x + 1 * (seg_w + seg_gap), y0, seg_w, item_h, "60FPS")
    menu_buttons["fpsmax"] = Button(menu_x + 2 * (seg_w + seg_gap), y0, seg_w, item_h, "MAX")

    # Gain toggle (full width)
    gain_y = y0 + (item_h + gap)
    menu_buttons["gain"] = Button(menu_x, gain_y, menu_w, item_h, f"GAIN: {button_state.gain_mode}")

    # Colormap cycle (full width)
    colormap_y = gain_y + (item_h + gap)
    menu_buttons["colormap"] = Button(menu_x, colormap_y, menu_w, item_h, f"MAP: {button_state.colormap_mode}")

    # Camera toggle (full width)
    cam_y = colormap_y + (item_h + gap)
    menu_buttons["cam"] = Button(
        menu_x, cam_y, menu_w, item_h,
        "CAMERA: ON" if button_state.camera_enabled else "CAMERA: OFF"
    )

    # Source cycle (full width)
    src_y = cam_y + (item_h + gap)
    menu_buttons["source"] = Button(menu_x, src_y, menu_w, item_h, f"SOURCE: {button_state.source_mode}")

    # Debug toggle (full width)
    dbg_y = src_y + (item_h + gap)
    menu_buttons["debug"] = Button(
        menu_x, dbg_y, menu_w, item_h,
        "DEBUG: ON" if button_state.debug_enabled else "DEBUG: OFF"
    )

    # Segmented tools under DEBUG (SHOT | REC | PAUSE)
    tools_y = dbg_y + (item_h + gap)
    tool_gap = 6
    tool_w = (menu_w - 2 * tool_gap) // 3

    menu_buttons["shot"]  = Button(menu_x + 0 * (tool_w + tool_gap), tools_y, tool_w, item_h, "SHOT")
    menu_buttons["rec"]   = Button(menu_x + 1 * (tool_w + tool_gap), tools_y, tool_w, item_h, "REC")
    menu_buttons["pause"] = Button(menu_x + 2 * (tool_w + tool_gap), tools_y, tool_w, item_h, "PAUSE")

    gallery_y = tools_y + (item_h + gap)
    menu_buttons["gallery"] = Button(menu_x, gallery_y, menu_w, item_h, "GALLERY")


def update_button_states(mx: int, my: int) -> None:
    for b in buttons.values():
        b.is_hovered = b.contains(mx, my)

    if "menu" in menu_buttons:
        menu_buttons["menu"].is_hovered = menu_buttons["menu"].contains(mx, my)

    keys = ("fps30", "fps60", "fpsmax", "gain", "colormap", "cam", "source", "debug", "shot", "rec", "pause", "gallery")

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
    
    # Dynamically position menu button at bottom of actual frame
    menu_btn = menu_buttons["menu"]
    actual_frame_height = frame.shape[0]
    menu_btn.y = actual_frame_height - menu_btn.h - 5  # 5px from bottom
    
    # Draw transparent menu button (similar to HUD pills)
    x, y, w, h = menu_btn.x, menu_btn.y, menu_btn.w, menu_btn.h
    
    # Semi-transparent background
    roi = frame[y:y+h, x:x+w]
    overlay = np.empty_like(roi)
    bg_color = (40, 200, 60) if menu_btn.is_active else ((85, 85, 85) if menu_btn.is_hovered else (60, 60, 60))
    overlay[:] = bg_color
    # Use higher opacity for active state to make green more obvious
    alpha = 0.25 if menu_btn.is_active else 0.12
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)
    
    # Border - use green border for active state
    border_color = (80, 255, 100) if menu_btn.is_active else (230, 230, 230)
    cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 2, cv2.LINE_8)
    
    # Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thick = 1
    tw, th = cv2.getTextSize(menu_btn.text, font, scale, thick)[0]
    tx = x + (w - tw) // 2
    ty = y + (h + th) // 2
    cv2.putText(frame, menu_btn.text, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_8)

    if not button_state.menu_open:
        return

    # Draw dropdown above menu button (no background container)
    menu_buttons["fps30"].is_active = (button_state.fps_mode == "30")
    menu_buttons["fps60"].is_active = (button_state.fps_mode == "60")
    menu_buttons["fpsmax"].is_active = (button_state.fps_mode == "MAX")

    # GAIN: always active, but LOW=dark green, HIGH=bright green
    menu_buttons["gain"].is_active = True
    
    # MAP: always active (always green)
    menu_buttons["colormap"].is_active = True

    menu_buttons["cam"].is_active = button_state.camera_enabled
    
    menu_buttons["debug"].is_active = button_state.debug_enabled

    # Tool button actives reflect current state
    menu_buttons["rec"].is_active = button_state.is_recording
    menu_buttons["pause"].is_active = button_state.is_paused

    menu_buttons["gallery"].is_active = button_state.gallery_open

    # Draw all menu dropdown buttons with transparent style
    menu_buttons["fps30"].draw(frame, transparent=True)
    menu_buttons["fps60"].draw(frame, transparent=True)
    menu_buttons["fpsmax"].draw(frame, transparent=True)
    
    # GAIN: dark green for LOW, bright green for HIGH
    gain_color = (30, 100, 40) if button_state.gain_mode == "LOW" else (40, 200, 60)
    menu_buttons["gain"].draw(frame, transparent=True, active_color=gain_color)
    
    menu_buttons["colormap"].draw(frame, transparent=True)
    menu_buttons["cam"].draw(frame, transparent=True)
    menu_buttons["source"].draw(frame, transparent=True)
    menu_buttons["debug"].draw(frame, transparent=True)

    menu_buttons["shot"].draw(frame, transparent=True, icon_type="camera")
    menu_buttons["rec"].draw(frame, transparent=True, icon_type="rec")
    menu_buttons["pause"].draw(frame, transparent=True, icon_type="pause")
    
    # Don't draw gallery button if recording (replaced by timestamp)
    if not button_state.is_recording:
        menu_buttons["gallery"].draw(frame, transparent=True)


def draw_recording_timestamp(frame: np.ndarray, video_recorder: Optional[VideoRecorder]) -> None:
    """
    Draw recording timestamp replacing the GALLERY button position when recording.
    Shows elapsed time in MM:SS format, with red dot indicator.
    """
    if video_recorder is None or not video_recorder.is_recording:
        return

    if "menu" not in menu_buttons or "gallery" not in menu_buttons:
        return

    # Get elapsed time
    elapsed = video_recorder.get_elapsed_time()
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    # Position at GALLERY button location (replaces it during recording)
    if button_state.menu_open:
        gallery_btn = menu_buttons["gallery"]
        menu_x = gallery_btn.x
        menu_w = gallery_btn.w
        timestamp_y = gallery_btn.y
        timestamp_h = gallery_btn.h
    else:
        # Menu is closed: position ABOVE MENU button
        menu_x = menu_buttons["menu"].x
        menu_w = menu_buttons["menu"].w
        timestamp_h = 35
        timestamp_y = menu_buttons["menu"].y - timestamp_h - 10

    # Bounds check
    if timestamp_y < 0 or timestamp_y + timestamp_h > frame.shape[0]:
        return

    roi = frame[timestamp_y:timestamp_y+timestamp_h, menu_x:menu_x+menu_w]
    if roi.shape[1] != menu_w or roi.shape[0] != timestamp_h:
        return

    paused = video_recorder.is_paused
    key = (menu_w, timestamp_h, paused)

    hud = _REC_HUD_CACHE.get(key)
    if hud is None:
        hud = np.zeros((timestamp_h, menu_w, 3), dtype=np.uint8)
        bg_color = (30, 30, 30) if not paused else (40, 40, 60)
        hud[:] = bg_color

        # subtle “glass” top band (cheap)
        band_h = 10
        band = hud.copy()
        band[:band_h] = np.clip(band[:band_h].astype(np.int16) + 15, 0, 255).astype(np.uint8)
        cv2.addWeighted(band, 0.35, hud, 0.65, 0.0, hud)

        # border baked in
        border_color = (200, 200, 200) if not paused else (100, 100, 200)
        cv2.rectangle(hud, (0, 0), (menu_w - 1, timestamp_h - 1), border_color, 1, cv2.LINE_8)

        _REC_HUD_CACHE[key] = hud

    # Draw transparent overlay instead of cached hud (always visible)
    overlay = np.empty_like(roi)
    bg_color = (100, 0, 0) if not paused else (40, 40, 100)
    overlay[:] = bg_color
    cv2.addWeighted(overlay, 0.25, roi, 0.75, 0.0, dst=roi)

    # Draw border
    border_color = (255, 50, 50) if not paused else (100, 100, 255)
    cv2.rectangle(frame, (menu_x, timestamp_y), (menu_x + menu_w, timestamp_y + timestamp_h),
              border_color, 2, cv2.LINE_8)

    # Draw recording indicator (red dot or paused icon)
    indicator_x = menu_x + 10
    indicator_y = timestamp_y + timestamp_h // 2

    if paused:
        bar_w = 3
        bar_h = 10
        bar_gap = 4
        cv2.rectangle(frame,
                    (indicator_x, indicator_y - bar_h//2),
                    (indicator_x + bar_w, indicator_y + bar_h//2),
                    (100, 150, 255), -1, cv2.LINE_8)
        cv2.rectangle(frame,
                    (indicator_x + bar_w + bar_gap, indicator_y - bar_h//2),
                    (indicator_x + 2*bar_w + bar_gap, indicator_y + bar_h//2),
                    (100, 150, 255), -1, cv2.LINE_8)
    else:
        pulse = int((time.time() * 2) % 2)
        if pulse:
            cv2.circle(frame, (indicator_x + 5, indicator_y), 6, (0, 0, 255), -1, cv2.LINE_8)
            cv2.circle(frame, (indicator_x + 5, indicator_y), 6, (255, 255, 255), 1, cv2.LINE_8)

    # Draw timestamp text
    timestamp_text = f"{minutes:02d}:{seconds:02d}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thick = 1
    text_color = (255, 255, 255) if not video_recorder.is_paused else (150, 150, 255)

    if timestamp_text in _REC_TEXT_SIZE_CACHE:
        text_w, text_h = _REC_TEXT_SIZE_CACHE[timestamp_text]
    else:
        (text_w, text_h), _ = cv2.getTextSize(timestamp_text, font, font_scale, font_thick)
        _REC_TEXT_SIZE_CACHE[timestamp_text] = (text_w, text_h)

    text_color = (255, 255, 255) if not paused else (150, 150, 255)

    text_x = menu_x + (menu_w - text_w) // 2 + 10
    text_y = timestamp_y + (timestamp_h + text_h) // 2

    cv2.putText(frame, timestamp_text, (text_x, text_y),
            font, font_scale, text_color, font_thick, cv2.LINE_8)


# ===============================================================
# Gallery view functions
# ===============================================================
def get_gallery_items(output_dir: Path) -> List[Tuple[Path, str, datetime]]:
    """
    Get all screenshots and videos from the output directory.
    Returns list of tuples: (filepath, type, modification_time)
    Sorted by modification time (newest first).
    """
    items = []

    if not output_dir.exists():
        return items

    # Get all screenshots (PNG files)
    for img_file in output_dir.glob("screenshot_*.png"):
        mtime = datetime.fromtimestamp(img_file.stat().st_mtime)
        items.append((img_file, "image", mtime))

    # Get all recordings (MP4 files)
    for vid_file in output_dir.glob("recording_*.mp4"):
        mtime = datetime.fromtimestamp(vid_file.stat().st_mtime)
        items.append((vid_file, "video", mtime))

    # Sort by modification time (newest first)
    items.sort(key=lambda x: x[2], reverse=True)

    return items


def draw_image_viewer(frame: np.ndarray, items: List[Tuple[Path, str, datetime]], output_dir: Optional[Path]) -> None:
    """
    Draw full-screen image viewer.
    """
    # Black background
    frame[:] = (0, 0, 0)
    
    if button_state.gallery_selected_item is None or button_state.gallery_selected_item >= len(items):
        return
    
    filepath, item_type, mtime = items[button_state.gallery_selected_item]
    
    # Load image
    img = cv2.imread(str(filepath))
    if img is None:
        # Error message
        font = cv2.FONT_HERSHEY_SIMPLEX
        msg = "Failed to load image"
        (msg_w, msg_h), _ = cv2.getTextSize(msg, font, 0.7, 1)
        cv2.putText(frame, msg, ((frame.shape[1] - msg_w) // 2, frame.shape[0] // 2),
                   font, 0.7, (150, 150, 150), 1, cv2.LINE_AA)
    else:
        # Scale image to fit screen while maintaining aspect ratio
        h, w = img.shape[:2]
        frame_h, frame_w = frame.shape[:2]
        
        # Calculate scaling factor
        scale = min(frame_w / w, frame_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Center image
        x_offset = (frame_w - new_w) // 2
        y_offset = (frame_h - new_h) // 2
        
        frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
    
    # Draw controls overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Back button (top-left)
    back_btn_x = 20
    back_btn_y = 20
    back_btn_w = 100
    back_btn_h = 40
    
    if "gallery_back" not in menu_buttons:
        menu_buttons["gallery_back"] = Button(back_btn_x, back_btn_y, back_btn_w, back_btn_h, "< BACK")
    else:
        menu_buttons["gallery_back"].x = back_btn_x
        menu_buttons["gallery_back"].y = back_btn_y
        menu_buttons["gallery_back"].w = back_btn_w
        menu_buttons["gallery_back"].h = back_btn_h
    
    menu_buttons["gallery_back"].draw(frame, transparent=True)
    
    # Navigation arrows
    if button_state.gallery_selected_item > 0:
        # Previous button
        prev_btn_x = 20
        prev_btn_y = frame.shape[0] // 2 - 30
        prev_btn_w = 60
        prev_btn_h = 60
        
        if "gallery_prev" not in menu_buttons:
            menu_buttons["gallery_prev"] = Button(prev_btn_x, prev_btn_y, prev_btn_w, prev_btn_h, "<")
        else:
            menu_buttons["gallery_prev"].x = prev_btn_x
            menu_buttons["gallery_prev"].y = prev_btn_y
            menu_buttons["gallery_prev"].w = prev_btn_w
            menu_buttons["gallery_prev"].h = prev_btn_h
        
        menu_buttons["gallery_prev"].draw(frame, transparent=True)
    
    if button_state.gallery_selected_item < len(items) - 1:
        # Next button
        next_btn_x = frame.shape[1] - 80
        next_btn_y = frame.shape[0] // 2 - 30
        next_btn_w = 60
        next_btn_h = 60
        
        if "gallery_next" not in menu_buttons:
            menu_buttons["gallery_next"] = Button(next_btn_x, next_btn_y, next_btn_w, next_btn_h, ">")
        else:
            menu_buttons["gallery_next"].x = next_btn_x
            menu_buttons["gallery_next"].y = next_btn_y
            menu_buttons["gallery_next"].w = next_btn_w
            menu_buttons["gallery_next"].h = next_btn_h
        
        menu_buttons["gallery_next"].draw(frame, transparent=True)
    
    # Filename at bottom
    filename = filepath.name
    (text_w, text_h), _ = cv2.getTextSize(filename, font, 0.6, 1)
    text_x = (frame.shape[1] - text_w) // 2
    text_y = frame.shape[0] - 20
    
    # Semi-transparent background for text
    cv2.rectangle(frame, (text_x - 10, text_y - text_h - 10), 
                 (text_x + text_w + 10, text_y + 10), (0, 0, 0), -1)
    cv2.putText(frame, filename, (text_x, text_y), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def draw_video_viewer(frame: np.ndarray, items: List[Tuple[Path, str, datetime]], output_dir: Optional[Path]) -> None:
    """
    Draw video player with controls.
    """
    # Black background
    frame[:] = (0, 0, 0)
    
    if button_state.gallery_selected_item is None or button_state.gallery_selected_item >= len(items):
        return
    
    filepath, item_type, mtime = items[button_state.gallery_selected_item]
    
    # Open video
    cap = cv2.VideoCapture(str(filepath))
    
    if not cap.isOpened():
        # Error message
        font = cv2.FONT_HERSHEY_SIMPLEX
        msg = "Failed to load video"
        (msg_w, msg_h), _ = cv2.getTextSize(msg, font, 0.7, 1)
        cv2.putText(frame, msg, ((frame.shape[1] - msg_w) // 2, frame.shape[0] // 2),
                   font, 0.7, (150, 150, 150), 1, cv2.LINE_AA)
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Handle auto-play
    if button_state.gallery_video_playing:
        button_state.gallery_video_frame_idx += 1
        if button_state.gallery_video_frame_idx >= total_frames:
            button_state.gallery_video_frame_idx = 0  # Loop
    
    # Clamp frame index
    button_state.gallery_video_frame_idx = max(0, min(button_state.gallery_video_frame_idx, total_frames - 1))
    
    # Seek to current frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, button_state.gallery_video_frame_idx)
    ret, vid_frame = cap.read()
    cap.release()
    
    if ret and vid_frame is not None:
        # Scale video to fit screen
        h, w = vid_frame.shape[:2]
        frame_h, frame_w = frame.shape[:2]
        
        # Leave space for controls at bottom
        controls_h = 100
        available_h = frame_h - controls_h
        
        scale = min(frame_w / w, available_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        vid_resized = cv2.resize(vid_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Center video
        x_offset = (frame_w - new_w) // 2
        y_offset = (available_h - new_h) // 2
        
        frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = vid_resized
    
    # Draw controls
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Back button
    back_btn_x = 20
    back_btn_y = 20
    back_btn_w = 100
    back_btn_h = 40
    
    if "gallery_back" not in menu_buttons:
        menu_buttons["gallery_back"] = Button(back_btn_x, back_btn_y, back_btn_w, back_btn_h, "< BACK")
    else:
        menu_buttons["gallery_back"].x = back_btn_x
        menu_buttons["gallery_back"].y = back_btn_y
        menu_buttons["gallery_back"].w = back_btn_w
        menu_buttons["gallery_back"].h = back_btn_h
    
    menu_buttons["gallery_back"].draw(frame, transparent=True)
    
    # Control panel at bottom
    controls_y = frame.shape[0] - 90
    cv2.rectangle(frame, (0, controls_y), (frame.shape[1], frame.shape[0]), (30, 30, 30), -1)
    
    # Play/Pause button
    play_btn_x = frame.shape[1] // 2 - 30
    play_btn_y = controls_y + 10
    play_btn_w = 60
    play_btn_h = 40
    play_text = "PAUSE" if button_state.gallery_video_playing else "PLAY"
    
    if "gallery_play" not in menu_buttons:
        menu_buttons["gallery_play"] = Button(play_btn_x, play_btn_y, play_btn_w, play_btn_h, play_text)
    else:
        menu_buttons["gallery_play"].x = play_btn_x
        menu_buttons["gallery_play"].y = play_btn_y
        menu_buttons["gallery_play"].w = play_btn_w
        menu_buttons["gallery_play"].h = play_btn_h
        menu_buttons["gallery_play"].text = play_text
    
    menu_buttons["gallery_play"].draw(frame, transparent=True)
    
    # Progress bar
    progress_x = 50
    progress_y = controls_y + 60
    progress_w = frame.shape[1] - 100
    progress_h = 10
    
    cv2.rectangle(frame, (progress_x, progress_y), (progress_x + progress_w, progress_y + progress_h),
                 (60, 60, 60), -1)
    
    # Progress fill
    if total_frames > 0:
        progress = button_state.gallery_video_frame_idx / total_frames
        fill_w = int(progress_w * progress)
        cv2.rectangle(frame, (progress_x, progress_y), (progress_x + fill_w, progress_y + progress_h),
                     (100, 200, 255), -1)
    
    # Store progress bar for click detection
    if "gallery_progress" not in menu_buttons:
        menu_buttons["gallery_progress"] = Button(progress_x, progress_y, progress_w, progress_h, "")
    else:
        menu_buttons["gallery_progress"].x = progress_x
        menu_buttons["gallery_progress"].y = progress_y
        menu_buttons["gallery_progress"].w = progress_w
        menu_buttons["gallery_progress"].h = progress_h
    
    # Time display
    current_time = button_state.gallery_video_frame_idx / fps if fps > 0 else 0
    total_time = total_frames / fps if fps > 0 else 0
    time_text = f"{int(current_time // 60):02d}:{int(current_time % 60):02d} / {int(total_time // 60):02d}:{int(total_time % 60):02d}"
    
    (time_w, time_h), _ = cv2.getTextSize(time_text, font, 0.5, 1)
    time_x = frame.shape[1] - time_w - 20
    time_y = controls_y + 35
    cv2.putText(frame, time_text, (time_x, time_y), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Filename
    filename = filepath.name
    (text_w, text_h), _ = cv2.getTextSize(filename, font, 0.5, 1)
    cv2.putText(frame, filename, (20, controls_y + 35), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def draw_gallery_view(frame: np.ndarray, output_dir: Optional[Path]) -> None:
    """
    Draw the gallery view showing all saved screenshots and videos.
    Displays items in a grid layout with thumbnails.
    Supports scrolling and clickable thumbnails.
    """
    if output_dir is None:
        return

    # Get all gallery items
    items = get_gallery_items(output_dir)
    
    # Check if we're in viewer mode
    if button_state.gallery_viewer_mode == "image":
        draw_image_viewer(frame, items, output_dir)
        return
    elif button_state.gallery_viewer_mode == "video":
        draw_video_viewer(frame, items, output_dir)
        return

    # Grid view mode
    # Dark semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)

    # Header
    header_h = 80
    cv2.rectangle(frame, (0, 0), (frame.shape[1], header_h), (25, 25, 25), -1)
    cv2.line(frame, (0, header_h), (frame.shape[1], header_h), (80, 80, 80), 2)

    # Title
    title = "GALLERY"
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 1.2
    title_thick = 2
    (title_w, title_h), _ = cv2.getTextSize(title, font, title_scale, title_thick)
    title_x = (frame.shape[1] - title_w) // 2
    title_y = (header_h + title_h) // 2 + 5
    cv2.putText(frame, title, (title_x, title_y), font, title_scale, (255, 255, 255), title_thick, cv2.LINE_AA)

    # Back button
    back_btn_x = 20
    back_btn_y = 20
    back_btn_w = 100
    back_btn_h = 40

    # Store back button for click detection
    if "gallery_back" not in menu_buttons:
        menu_buttons["gallery_back"] = Button(back_btn_x, back_btn_y, back_btn_w, back_btn_h, "< BACK")
    else:
        menu_buttons["gallery_back"].x = back_btn_x
        menu_buttons["gallery_back"].y = back_btn_y
        menu_buttons["gallery_back"].w = back_btn_w
        menu_buttons["gallery_back"].h = back_btn_h

    menu_buttons["gallery_back"].draw(frame, transparent=True)

    if not items:
        # No items message
        msg = "No captures yet. Use SHOT or REC to create content."
        msg_scale = 0.7
        msg_thick = 1
        (msg_w, msg_h), _ = cv2.getTextSize(msg, font, msg_scale, msg_thick)
        msg_x = (frame.shape[1] - msg_w) // 2
        msg_y = frame.shape[0] // 2
        cv2.putText(frame, msg, (msg_x, msg_y), font, msg_scale, (150, 150, 150), msg_thick, cv2.LINE_AA)
        return

    # Grid layout parameters
    margin = 20
    grid_start_y = header_h + margin
    grid_start_x = margin

    thumb_w = 280
    thumb_h = 180
    gap = 15

    cols = (frame.shape[1] - 2 * margin + gap) // (thumb_w + gap)
    cols = max(1, cols)
    
    # Calculate total content height
    total_rows = (len(items) + cols - 1) // cols  # Ceiling division
    item_height = thumb_h + gap + 50  # Thumbnail + gap + label space
    total_content_height = grid_start_y + (total_rows * item_height)
    footer_space = 40
    
    # Calculate max scroll (how much content extends beyond visible area)
    visible_height = frame.shape[0] - header_h - footer_space
    max_scroll = max(0, total_content_height - frame.shape[0] + footer_space)
    
    # Clamp scroll offset to valid range
    button_state.gallery_scroll_offset = max(0, min(button_state.gallery_scroll_offset, max_scroll))

    # Store thumbnail positions for click detection
    if not hasattr(button_state, 'gallery_thumbnail_rects'):
        button_state.gallery_thumbnail_rects = []
    button_state.gallery_thumbnail_rects.clear()

    # Apply scroll offset
    scroll_offset = button_state.gallery_scroll_offset

    # Draw grid of thumbnails
    for idx, (filepath, item_type, mtime) in enumerate(items):
        row = idx // cols
        col = idx % cols

        x = grid_start_x + col * (thumb_w + gap)
        y = grid_start_y + row * (thumb_h + gap + 50) - scroll_offset  # Apply scroll

        # Skip if completely off-screen
        if y + thumb_h + 50 < header_h or y > frame.shape[0]:
            continue

        # Store thumbnail rect for click detection
        button_state.gallery_thumbnail_rects.append({
            'idx': idx,
            'x': x,
            'y': y,
            'w': thumb_w,
            'h': thumb_h,
            'filepath': filepath,
            'type': item_type
        })

        # Draw thumbnail background
        cv2.rectangle(frame, (x, y), (x + thumb_w, y + thumb_h), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, y), (x + thumb_w, y + thumb_h), (100, 100, 100), 2, cv2.LINE_AA)

        # Load and draw thumbnail
        try:
            if item_type == "image":
                img = cv2.imread(str(filepath))
                if img is not None:
                    # Resize to fit thumbnail
                    img_resized = cv2.resize(img, (thumb_w - 4, thumb_h - 4), interpolation=cv2.INTER_AREA)
                    # Clip to visible area
                    if y + 2 >= header_h and y + thumb_h - 2 <= frame.shape[0]:
                        frame[y+2:y+thumb_h-2, x+2:x+thumb_w-2] = img_resized
            elif item_type == "video":
                # For videos, show first frame as thumbnail
                cap = cv2.VideoCapture(str(filepath))
                ret, vid_frame = cap.read()
                cap.release()

                if ret and vid_frame is not None:
                    vid_resized = cv2.resize(vid_frame, (thumb_w - 4, thumb_h - 4), interpolation=cv2.INTER_AREA)
                    # Clip to visible area
                    if y + 2 >= header_h and y + thumb_h - 2 <= frame.shape[0]:
                        frame[y+2:y+thumb_h-2, x+2:x+thumb_w-2] = vid_resized

                    # Draw play icon overlay for videos
                    center_x = x + thumb_w // 2
                    center_y = y + thumb_h // 2
                    play_size = 30

                    if center_y - play_size >= 0 and center_y + play_size <= frame.shape[0]:
                        # Semi-transparent circle
                        overlay_roi = frame[center_y-play_size:center_y+play_size,
                                           center_x-play_size:center_x+play_size].copy()
                        cv2.circle(overlay_roi, (play_size, play_size), play_size, (0, 0, 0), -1)
                        cv2.addWeighted(overlay_roi, 0.6,
                                      frame[center_y-play_size:center_y+play_size,
                                           center_x-play_size:center_x+play_size], 0.4, 0,
                                      frame[center_y-play_size:center_y+play_size,
                                           center_x-play_size:center_x+play_size])

                        # White play triangle
                        pts = np.array([
                            [center_x - 10, center_y - 15],
                            [center_x - 10, center_y + 15],
                            [center_x + 15, center_y]
                        ], np.int32)
                        cv2.fillPoly(frame, [pts], (255, 255, 255), cv2.LINE_AA)
        except Exception as e:
            # Draw error placeholder
            cv2.putText(frame, "Error", (x + 10, y + thumb_h // 2),
                       font, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

        # Draw label below thumbnail
        label_y = y + thumb_h + 20
        if label_y >= header_h and label_y <= frame.shape[0]:
            filename = filepath.name

            # Truncate filename if too long
            if len(filename) > 30:
                filename = filename[:27] + "..."

            # File type indicator
            type_icon = "[IMG]" if item_type == "image" else "[VID]"
            type_color = (100, 200, 100) if item_type == "image" else (100, 150, 255)

            cv2.putText(frame, type_icon, (x, label_y), font, 0.45, type_color, 1, cv2.LINE_AA)

            # Filename
            cv2.putText(frame, filename, (x, label_y + 20), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    # Footer with count and scroll hint
    footer_text = f"Total: {len(items)} items ({sum(1 for _, t, _ in items if t == 'image')} images, {sum(1 for _, t, _ in items if t == 'video')} videos) | Swipe to scroll"
    footer_y = frame.shape[0] - 15
    cv2.putText(frame, footer_text, (margin, footer_y), font, 0.5, (150, 150, 150), 1, cv2.LINE_AA)


# ===============================================================
# Click handlers
# ===============================================================
def handle_gallery_click(x: int, y: int, output_dir: Optional[Path]) -> bool:
    """
    Handle clicks in gallery view.
    Returns True if click was handled in gallery, False otherwise.
    """
    if not button_state.gallery_open:
        return False

    # Check back button
    if "gallery_back" in menu_buttons and menu_buttons["gallery_back"].contains(x, y):
        if button_state.gallery_viewer_mode == "grid":
            # Close gallery
            button_state.gallery_open = False
            button_state.gallery_scroll_offset = 0
            button_state.gallery_selected_item = None
        else:
            # Return to grid view
            button_state.gallery_viewer_mode = "grid"
            button_state.gallery_video_playing = False
            button_state.gallery_video_frame_idx = 0
        return True
    
    # Handle viewer mode clicks
    if button_state.gallery_viewer_mode == "image":
        # Previous button
        if "gallery_prev" in menu_buttons and menu_buttons["gallery_prev"].contains(x, y):
            if button_state.gallery_selected_item > 0:
                button_state.gallery_selected_item -= 1
                # Skip to previous image if current is video
                items = get_gallery_items(output_dir) if output_dir else []
                while button_state.gallery_selected_item > 0 and items[button_state.gallery_selected_item][1] != "image":
                    button_state.gallery_selected_item -= 1
            return True
        
        # Next button
        if "gallery_next" in menu_buttons and menu_buttons["gallery_next"].contains(x, y):
            items = get_gallery_items(output_dir) if output_dir else []
            if button_state.gallery_selected_item < len(items) - 1:
                button_state.gallery_selected_item += 1
                # Skip to next image if current is video
                while button_state.gallery_selected_item < len(items) - 1 and items[button_state.gallery_selected_item][1] != "image":
                    button_state.gallery_selected_item += 1
            return True
    
    elif button_state.gallery_viewer_mode == "video":
        # Play/Pause button
        if "gallery_play" in menu_buttons and menu_buttons["gallery_play"].contains(x, y):
            button_state.gallery_video_playing = not button_state.gallery_video_playing
            return True
        
        # Progress bar click (seek)
        if "gallery_progress" in menu_buttons and menu_buttons["gallery_progress"].contains(x, y):
            items = get_gallery_items(output_dir) if output_dir else []
            if button_state.gallery_selected_item is not None and button_state.gallery_selected_item < len(items):
                filepath = items[button_state.gallery_selected_item][0]
                cap = cv2.VideoCapture(str(filepath))
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    
                    # Calculate clicked position
                    progress_btn = menu_buttons["gallery_progress"]
                    click_pos = (x - progress_btn.x) / progress_btn.w
                    click_pos = max(0.0, min(1.0, click_pos))
                    
                    button_state.gallery_video_frame_idx = int(total_frames * click_pos)
                    button_state.gallery_video_playing = False  # Pause on seek
            return True
    
    # Handle grid view clicks (thumbnail selection)
    if button_state.gallery_viewer_mode == "grid":
        if hasattr(button_state, 'gallery_thumbnail_rects'):
            for thumb in button_state.gallery_thumbnail_rects:
                if (thumb['x'] <= x <= thumb['x'] + thumb['w'] and
                    thumb['y'] <= y <= thumb['y'] + thumb['h']):
                    # Thumbnail clicked
                    button_state.gallery_selected_item = thumb['idx']
                    if thumb['type'] == "image":
                        button_state.gallery_viewer_mode = "image"
                    else:
                        button_state.gallery_viewer_mode = "video"
                        button_state.gallery_video_playing = False
                        button_state.gallery_video_frame_idx = 0
                    return True

    return True  # Consume all clicks when gallery is open


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

    # Colormap cycle (MAGMA -> JET -> TURBO -> INFERNO)
    if "colormap" in menu_buttons and menu_buttons["colormap"].contains(x, y):
        colormaps = ["MAGMA", "JET", "TURBO", "INFERNO"]
        cur = button_state.colormap_mode
        try:
            i = colormaps.index(cur)
        except ValueError:
            i = 0
        button_state.colormap_mode = colormaps[(i + 1) % len(colormaps)]
        menu_buttons["colormap"].text = f"MAP: {button_state.colormap_mode}"
        return video_recorder

    # CAMERA toggle
    if "cam" in menu_buttons and menu_buttons["cam"].contains(x, y):
        button_state.camera_enabled = not button_state.camera_enabled
        menu_buttons["cam"].text = "CAMERA: ON" if button_state.camera_enabled else "CAMERA: OFF"
        return video_recorder

    # SOURCE cycle
    if "source" in menu_buttons and menu_buttons["source"].contains(x, y):
        modes = list(SOURCE_MODES)
        cur = button_state.source_mode
        try:
            i = modes.index(cur)
        except ValueError:
            i = modes.index(SOURCE_DEFAULT)
        button_state.source_mode = modes[(i + 1) % len(modes)]
        menu_buttons["source"].text = f"SOURCE: {button_state.source_mode}"
        return video_recorder

    # DEBUG toggle
    if "debug" in menu_buttons and menu_buttons["debug"].contains(x, y):
        button_state.debug_enabled = not button_state.debug_enabled
        menu_buttons["debug"].text = "DEBUG: ON" if button_state.debug_enabled else "DEBUG: OFF"
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
                menu_buttons["rec"].text = "STOP"  # Change text to STOP
            else:
                button_state.is_recording = False
        else:
            video_recorder.stop_recording()
            button_state.is_paused = False
            menu_buttons["rec"].text = "REC"  # Change text back to REC
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

    # GALLERY
    if "gallery" in menu_buttons and menu_buttons["gallery"].contains(x, y):
        button_state.gallery_open = True
        button_state.menu_open = False  # Close menu when opening gallery
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

    if "source" in buttons and buttons["source"].contains(x, y):
        modes = list(SOURCE_MODES)  # ("SIM", "SPI_LOOPBACK", "SPI_HW")
        cur = button_state.source_mode
        try:
            i = modes.index(cur)
        except ValueError:
            i = modes.index(SOURCE_DEFAULT)

        button_state.source_mode = modes[(i + 1) % len(modes)]
        buttons["source"].text = f"Source: {button_state.source_mode}"
        return video_recorder


    if "debug" in buttons and buttons["debug"].contains(x, y):
        button_state.debug_enabled = not button_state.debug_enabled
        buttons["debug"].is_active = button_state.debug_enabled
        buttons["debug"].text = "DEBUG: ON" if button_state.debug_enabled else "DEBUG: OFF"
        return video_recorder

    return video_recorder
