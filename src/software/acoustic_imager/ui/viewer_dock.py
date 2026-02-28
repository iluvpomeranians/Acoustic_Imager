"""
Viewer dock: shared bottom chrome for image/video gallery viewer.

Back button, dock with title/metadata/trash, and (for video) play + progress.
Taller dock with optional green click feedback for nav/play buttons.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional
import time

import cv2
import numpy as np

from .button import menu_buttons, Button
from ..state import button_state

# Taller dock for clearer layout
VIEWER_DOCK_HEIGHT = 105

# Green feedback (match main/HUD active green)
FEEDBACK_GREEN_BG = (40, 200, 60)
FEEDBACK_GREEN_BORDER = (80, 255, 100)
FEEDBACK_DURATION_S = 0.28


def _format_mtime(mtime: Optional[datetime]) -> str:
    """Short date/time for dock metadata. Returns '' if mtime is None."""
    if mtime is None:
        return ""
    try:
        return mtime.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


def _format_filesize(size_bytes: int) -> str:
    """Human-readable file size for dock metadata."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def trigger_viewer_button_feedback(button_key: str) -> None:
    """Call when user clicks back, prev, next, or play so dock can draw green flash."""
    button_state.viewer_dock_feedback_button = button_key
    button_state.viewer_dock_feedback_time = time.time()


def draw_viewer_back_button_on_top(frame: np.ndarray) -> None:
    """Draw the back button on top of viewer content so it is never covered. Call after drawing cards/arrows."""
    back_btn_x, back_btn_y = 20, 20
    back_btn_w, back_btn_h = 50, 40
    if "gallery_back" not in menu_buttons:
        menu_buttons["gallery_back"] = Button(back_btn_x, back_btn_y, back_btn_w, back_btn_h, "")
    else:
        menu_buttons["gallery_back"].x = back_btn_x
        menu_buttons["gallery_back"].y = back_btn_y
        menu_buttons["gallery_back"].w = back_btn_w
        menu_buttons["gallery_back"].h = back_btn_h
    menu_buttons["gallery_back"].draw(frame, transparent=True, icon_type="back")
    draw_viewer_button_feedback(
        frame, "gallery_back",
        back_btn_x, back_btn_y, back_btn_w, back_btn_h,
    )


def draw_viewer_button_feedback(frame: np.ndarray, button_key: str, x: int, y: int, w: int, h: int) -> None:
    """If this button was recently clicked, draw a short green flash overlay (same green as main)."""
    if getattr(button_state, "viewer_dock_feedback_button", "") != button_key:
        return
    elapsed = time.time() - getattr(button_state, "viewer_dock_feedback_time", 0.0)
    if elapsed > FEEDBACK_DURATION_S:
        return
    fade = 1.0 - (elapsed / FEEDBACK_DURATION_S)
    alpha = 0.35 * fade
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame.shape[1], x + w)
    y1 = min(frame.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    overlay = np.empty_like(roi)
    overlay[:] = FEEDBACK_GREEN_BG
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)
    border_alpha = 0.7 * fade
    if border_alpha > 0.05:
        cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), FEEDBACK_GREEN_BORDER, 2, cv2.LINE_AA)


def draw_viewer_chrome(
    frame: np.ndarray,
    filepath: Path,
    item_type: str,
    mtime: datetime,
    is_video: bool = False,
    total_frames: int = 0,
    fps: float = 0,
    current_idx: int = 0,
    play_text: str = "PLAY",
) -> int:
    """
    Draw shared viewer chrome: back button + bottom dock with filename, trash, metadata.
    For video, also draws play/pause and progress bar in the same dock.
    Returns controls_y (top of dock, content area bottom).
    """
    frame_h, frame_w = frame.shape[:2]
    controls_y = frame_h - VIEWER_DOCK_HEIGHT
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Back button is drawn on top after content (see draw_viewer_back_button_on_top)

    # Dock background
    cv2.rectangle(frame, (0, controls_y), (frame_w, frame_h), (30, 30, 30), -1)
    cv2.line(frame, (0, controls_y), (frame_w, controls_y), (70, 70, 70), 1)

    pad = 14
    row0_y = controls_y + 28
    font_scale_title = 0.58

    # File size for metadata
    try:
        size_str = _format_filesize(filepath.stat().st_size)
    except Exception:
        size_str = ""

    # Fixed left-to-right: [VID]/[IMG] | filename | metadata | delete
    type_label = "[VID]" if item_type == "video" else "[IMG]"
    type_color = (100, 150, 255) if item_type == "video" else (100, 200, 100)
    font_scale_meta = 0.48
    (type_w, _), _ = cv2.getTextSize(type_label, font, 0.6, 1)
    type_x = pad

    # Trash: fixed at right, wider, aligned to text row
    trash_w, trash_h = 58, 32
    trash_x = frame_w - pad - trash_w
    trash_y = row0_y - trash_h // 2  # same row as type/filename/metadata text
    if "gallery_delete" not in menu_buttons:
        menu_buttons["gallery_delete"] = Button(trash_x, trash_y, trash_w, trash_h, "")
    else:
        menu_buttons["gallery_delete"].x = trash_x
        menu_buttons["gallery_delete"].y = trash_y
        menu_buttons["gallery_delete"].w = trash_w
        menu_buttons["gallery_delete"].h = trash_h
    menu_buttons["gallery_delete"].is_active = True
    menu_buttons["gallery_delete"].draw(frame, transparent=True, active_color=(0, 0, 200), icon_type="trash")

    # Build metadata string (mtime, size, video time) – use ASCII separator to avoid "??" from · in font
    mtime_str = _format_mtime(mtime) if mtime is not None else ""
    meta_parts = []
    if mtime_str:
        meta_parts.append(mtime_str)
    if size_str:
        meta_parts.append(size_str)
    if is_video and total_frames > 0 and fps > 0:
        cur_s = current_idx / fps
        total_s = total_frames / fps
        meta_parts.append(f"{int(cur_s // 60):02d}:{int(cur_s % 60):02d} / {int(total_s // 60):02d}:{int(total_s % 60):02d}")
    meta_str = "  |  ".join(meta_parts)

    fn_x = type_x + type_w + 10
    gap = 10
    # Measure metadata first; give it the space it needs, filename gets the rest (only truncate when necessary)
    (meta_w, _), _ = cv2.getTextSize(meta_str, font, font_scale_meta, 1)
    meta_x = trash_x - gap - meta_w
    filename_max_w = meta_x - 8 - fn_x
    # If metadata is so long it would overlap filename, truncate metadata to free space
    min_filename_w = 60
    if filename_max_w < min_filename_w:
        max_meta_w = trash_x - gap - fn_x - 8 - min_filename_w
        if max_meta_w > 0 and meta_w > max_meta_w:
            for n in range(len(meta_str) - 1, 0, -1):
                candidate = meta_str[:n] + "..."
                (w, _), _ = cv2.getTextSize(candidate, font, font_scale_meta, 1)
                if w <= max_meta_w:
                    meta_str = candidate
                    meta_w = w
                    break
            meta_x = trash_x - gap - meta_w
            filename_max_w = meta_x - 8 - fn_x

    # 1) Type label leftmost
    cv2.putText(frame, type_label, (type_x, row0_y), font, 0.6, type_color, 1, cv2.LINE_AA)

    # 2) Filename next to type (truncate only if needed)
    filename = filepath.name
    (fn_w, _), _ = cv2.getTextSize(filename, font, font_scale_title, 1)
    if fn_w > filename_max_w and filename_max_w > 20:
        for n in range(len(filename), 0, -1):
            candidate = filename[:n] + "..."
            (w, _), _ = cv2.getTextSize(candidate, font, font_scale_title, 1)
            if w <= filename_max_w:
                filename = candidate
                break
        else:
            filename = "..."
    cv2.putText(frame, filename, (fn_x, row0_y), font, font_scale_title, (240, 240, 240), 1, cv2.LINE_AA)

    # 3) Metadata right-aligned before delete
    cv2.putText(frame, meta_str, (meta_x, row0_y), font, font_scale_meta, (170, 170, 170), 1, cv2.LINE_AA)

    if is_video:
        # Second row: play/pause + progress bar
        play_btn_w, play_btn_h = 80, 34
        play_btn_x = pad
        play_btn_y = controls_y + 52
        if "gallery_play" not in menu_buttons:
            menu_buttons["gallery_play"] = Button(play_btn_x, play_btn_y, play_btn_w, play_btn_h, play_text)
        else:
            menu_buttons["gallery_play"].x = play_btn_x
            menu_buttons["gallery_play"].y = play_btn_y
            menu_buttons["gallery_play"].w = play_btn_w
            menu_buttons["gallery_play"].h = play_btn_h
            menu_buttons["gallery_play"].text = play_text
        menu_buttons["gallery_play"].draw(frame, transparent=True)
        draw_viewer_button_feedback(
            frame, "gallery_play",
            play_btn_x, play_btn_y, play_btn_w, play_btn_h,
        )

        progress_x = play_btn_x + play_btn_w + 12
        progress_y = controls_y + 58
        progress_w = frame_w - progress_x - pad
        progress_h_visual = 8
        progress_h_hit = 52  # larger hit area for touch
        progress_y_hit = progress_y - (progress_h_hit - progress_h_visual) // 2
        cv2.rectangle(frame, (progress_x, progress_y), (progress_x + progress_w, progress_y + progress_h_visual),
                     (60, 60, 60), -1)
        if total_frames > 0:
            fill = int(progress_w * (current_idx / total_frames))
            cv2.rectangle(frame, (progress_x, progress_y), (progress_x + fill, progress_y + progress_h_visual),
                         (100, 200, 255), -1)
        if "gallery_progress" not in menu_buttons:
            menu_buttons["gallery_progress"] = Button(progress_x, progress_y_hit, progress_w, progress_h_hit, "")
        else:
            menu_buttons["gallery_progress"].x = progress_x
            menu_buttons["gallery_progress"].y = progress_y_hit
            menu_buttons["gallery_progress"].w = progress_w
            menu_buttons["gallery_progress"].h = progress_h_hit

    return controls_y
