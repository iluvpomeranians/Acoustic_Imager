"""
MENU dropdown and recording timestamp HUD.
"""

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from . import ui_cache
from .button import menu_buttons
from ..state import button_state
from .video_recorder import VideoRecorder
from ..config import BUTTON_ALPHA


def draw_menu(frame: np.ndarray) -> None:
    if "menu" not in menu_buttons:
        return

    menu_buttons["menu"].is_active = button_state.menu_open

    menu_btn = menu_buttons["menu"]
    actual_frame_height = frame.shape[0]
    menu_btn.y = actual_frame_height - menu_btn.h - 5

    x, y, w, h = menu_btn.x, menu_btn.y, menu_btn.w, menu_btn.h

    roi = frame[y:y+h, x:x+w]
    overlay = np.empty_like(roi)
    bg_color = (40, 200, 60) if menu_btn.is_active else (40, 40, 40)
    overlay[:] = bg_color
    alpha = BUTTON_ALPHA
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)

    border_color = (80, 255, 100) if menu_btn.is_active else (255, 255, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 2, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thick = 1
    tw, th = cv2.getTextSize(menu_btn.text, font, scale, thick)[0]
    tx = x + (w - tw) // 2
    ty = y + (h + th) // 2
    # cv2.putText(frame, menu_btn.text, (tx, ty), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(frame, menu_btn.text, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    if not button_state.menu_open:
        return

    menu_buttons["fps30"].is_active = (button_state.fps_mode == "30")
    menu_buttons["fps60"].is_active = (button_state.fps_mode == "60")
    menu_buttons["fpsmax"].is_active = (button_state.fps_mode == "MAX")

    menu_buttons["gain"].is_active = True
    menu_buttons["colormap"].is_active = True
    menu_buttons["cam"].is_active = button_state.camera_enabled
    menu_buttons["source"].is_active = True
    menu_buttons["debug"].is_active = button_state.debug_enabled
    menu_buttons["rec"].is_active = button_state.is_recording
    menu_buttons["gallery"].is_active = button_state.gallery_open

    menu_buttons["fps30"].draw(frame, transparent=True)
    menu_buttons["fps60"].draw(frame, transparent=True)
    menu_buttons["fpsmax"].draw(frame, transparent=True)

    gain_color = (40, 200, 60)
    menu_buttons["gain"].draw(frame, transparent=True, active_color=gain_color)

    menu_buttons["colormap"].draw(frame, transparent=True)
    menu_buttons["cam"].draw(frame, transparent=True)
    menu_buttons["source"].draw(frame, transparent=True)
    menu_buttons["debug"].draw(frame, transparent=True)

    menu_buttons["shot"].draw(frame, transparent=True, icon_type="camera")
    menu_buttons["rec"].draw(frame, transparent=True, icon_type="rec")

    if not button_state.is_recording:
        menu_buttons["gallery"].draw(frame, transparent=True)


def get_recording_timestamp_rect() -> Optional[Tuple[int, int, int, int]]:
    if not button_state.is_recording:
        return None

    if "menu" not in menu_buttons or "gallery" not in menu_buttons:
        return None

    RECORDING_BAR_HEIGHT = 50

    if button_state.menu_open:
        gallery_btn = menu_buttons["gallery"]
        menu_x = gallery_btn.x
        menu_w = gallery_btn.w
        timestamp_y = gallery_btn.y
        timestamp_h = RECORDING_BAR_HEIGHT
    else:
        menu_x = menu_buttons["menu"].x
        menu_w = menu_buttons["menu"].w
        timestamp_h = RECORDING_BAR_HEIGHT
        timestamp_y = menu_buttons["menu"].y - timestamp_h - 10

    return (menu_x, timestamp_y, menu_w, timestamp_h)


def draw_recording_timestamp(frame: np.ndarray, video_recorder: Optional[VideoRecorder]) -> None:
    if video_recorder is None or not video_recorder.is_recording:
        return

    rect = get_recording_timestamp_rect()
    if rect is None:
        return

    menu_x, timestamp_y, menu_w, timestamp_h = rect

    elapsed = video_recorder.get_elapsed_time()
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    if timestamp_y < 0 or timestamp_y + timestamp_h > frame.shape[0]:
        return

    roi = frame[timestamp_y:timestamp_y+timestamp_h, menu_x:menu_x+menu_w]
    if roi.shape[1] != menu_w or roi.shape[0] != timestamp_h:
        return

    paused = video_recorder.is_paused
    key = (menu_w, timestamp_h, paused)

    hud = ui_cache._REC_HUD_CACHE.get(key)
    if hud is None:
        hud = np.zeros((timestamp_h, menu_w, 3), dtype=np.uint8)
        bg_color = (30, 30, 30) if not paused else (40, 40, 60)
        hud[:] = bg_color

        band_h = 10
        band = hud.copy()
        band[:band_h] = np.clip(band[:band_h].astype(np.int16) + 15, 0, 255).astype(np.uint8)
        alpha = BUTTON_ALPHA
        cv2.addWeighted(band, alpha, hud, 1- alpha, 0.0, hud)

        border_color = (200, 200, 200) if not paused else (100, 100, 200)
        cv2.rectangle(hud, (0, 0), (menu_w - 1, timestamp_h - 1), border_color, 1, cv2.LINE_8)

        ui_cache._REC_HUD_CACHE[key] = hud

    overlay = np.empty_like(roi)
    bg_color = (100, 0, 0) if not paused else (0, 100, 150)
    overlay[:] = bg_color
    alpha = BUTTON_ALPHA
    cv2.addWeighted(overlay, alpha, roi, 1-alpha, 0.0, dst=roi)

    border_color = (255, 50, 50) if not paused else (0, 165, 255)
    cv2.rectangle(frame, (menu_x, timestamp_y), (menu_x + menu_w, timestamp_y + timestamp_h),
              border_color, 2, cv2.LINE_8)

    indicator_x = menu_x + 10
    indicator_y = timestamp_y + timestamp_h // 2

    if paused:
        bar_w = 3
        bar_h = 10
        bar_gap = 4
        cv2.rectangle(frame,
                    (indicator_x, indicator_y - bar_h//2),
                    (indicator_x + bar_w, indicator_y + bar_h//2),
                    (0, 200, 255), -1, cv2.LINE_8)
        cv2.rectangle(frame,
                    (indicator_x + bar_w + bar_gap, indicator_y - bar_h//2),
                    (indicator_x + 2*bar_w + bar_gap, indicator_y + bar_h//2),
                    (0, 200, 255), -1, cv2.LINE_8)
    else:
        pulse = int((time.time() * 2) % 2)
        if pulse:
            cv2.circle(frame, (indicator_x + 5, indicator_y), 6, (0, 0, 255), -1, cv2.LINE_8)
            cv2.circle(frame, (indicator_x + 5, indicator_y), 6, (255, 255, 255), 1, cv2.LINE_8)

    timestamp_text = f"{minutes:02d}:{seconds:02d}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thick = 1

    if timestamp_text in ui_cache._REC_TEXT_SIZE_CACHE:
        text_w, text_h = ui_cache._REC_TEXT_SIZE_CACHE[timestamp_text]
    else:
        (text_w, text_h), _ = cv2.getTextSize(timestamp_text, font, font_scale, font_thick)
        ui_cache._REC_TEXT_SIZE_CACHE[timestamp_text] = (text_w, text_h)

    text_color = (255, 255, 255) if not paused else (200, 230, 255)

    text_x = menu_x + (menu_w - text_w) // 2 + 10
    text_y = timestamp_y + (timestamp_h + text_h) // 2

    cv2.putText(frame, timestamp_text, (text_x, text_y),
            font, font_scale, text_color, font_thick, cv2.LINE_8)
