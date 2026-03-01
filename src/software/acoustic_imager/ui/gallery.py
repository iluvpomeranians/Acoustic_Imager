"""
Gallery view: grid, image/video viewers, delete modal.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import time
import os
import shutil

import cv2
import numpy as np

from . import ui_cache
from .button import menu_buttons, Button
from .viewer_dock import (
    draw_viewer_chrome,
    draw_viewer_back_button_on_top,
    draw_viewer_button_feedback,
    VIEWER_DOCK_HEIGHT as VIEWER_DOCK_H,
)
from ..state import button_state

# Rubber band at first/last item: stiff so no ghost card appears beyond the end
VIEWER_RUBBER_BAND_FACTOR = 0.35

# Carousel: card width and gap so next/prev are always visible (continuous peek)
VIEWER_CARD_WIDTH_RATIO = 0.88   # card 88% of viewport → ~6% peek each side, slight step-back, more fluid
VIEWER_CARD_GAP_RATIO = 0.02     # 2% gap between cards
VIEWER_SWIPE_SNAP_RATIO = 0.42   # snap to next/prev after dragging this fraction of slot width
VIEWER_AT_REST_EPSILON = 1.0     # when |offset| < this and not dragging/inertia, show only current (black borders)
# At first/last: show neighbor only when swiping toward it past this (px). During inertia never show neighbor (no ghost).
VIEWER_EDGE_PEEK_THRESHOLD_PX = 18

# Cache first frame of videos for carousel peek (path -> BGR frame)
_video_first_frame_cache: Dict[Path, np.ndarray] = {}


def _viewer_rubber_band_offset(offset: float, idx: int, n: int) -> float:
    """Apply rubber-band damping when dragging past first or last item."""
    if n <= 0:
        return 0.0
    if idx <= 0 and offset < 0:
        return offset * VIEWER_RUBBER_BAND_FACTOR
    if idx >= n - 1 and offset > 0:
        return offset * VIEWER_RUBBER_BAND_FACTOR
    return offset


def _viewer_at_rest() -> bool:
    """True when not swiping: no drag, no inertia, offset near zero. Show only current card with black borders."""
    off = getattr(button_state, "gallery_viewer_swipe_offset", 0.0)
    dragging = getattr(button_state, "gallery_viewer_swipe_dragging", False)
    inertia = getattr(button_state, "gallery_viewer_swipe_inertia_active", False)
    # Snap to full-width sooner: treat as at rest when offset has decayed to ~15px even if inertia still ticking
    if inertia and abs(off) < 15.0:
        return True
    return not dragging and not inertia and abs(off) < VIEWER_AT_REST_EPSILON


def _carousel_indices(idx: int, n: int) -> List[int]:
    """Indices to draw for carousel: current plus prev/next only when in range. No wrap."""
    out = [idx]
    if idx > 0:
        out.insert(0, idx - 1)
    if idx < n - 1:
        out.append(idx + 1)
    return out


def _blit_card_into_view(
    frame: np.ndarray,
    card_buf: np.ndarray,
    card_left: int,
    card_w: int,
    frame_w: int,
    available_h: int,
) -> None:
    """Blit the visible portion of a carousel card (card_w x available_h) into the viewport."""
    dst_x0 = max(0, card_left)
    dst_x1 = min(frame_w, card_left + card_w)
    if dst_x0 >= dst_x1:
        return
    src_x0 = dst_x0 - card_left
    src_x1 = dst_x1 - card_left
    view = frame[0:available_h, dst_x0:dst_x1]
    view[:] = card_buf[0:available_h, src_x0:src_x1]


def _make_image_card(img: np.ndarray, card_w: int, card_h: int) -> np.ndarray:
    """Render image scaled and centered into a (card_w x card_h) card buffer."""
    card = np.zeros((card_h, card_w, 3), dtype=np.uint8)
    card[:] = (20, 20, 20)
    h, w = img.shape[:2]
    scale = min(card_w / w, card_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x0 = (card_w - new_w) // 2
    y0 = (card_h - new_h) // 2
    card[y0 : y0 + new_h, x0 : x0 + new_w] = img_resized
    return card


def _get_video_first_frame(filepath: Path) -> Optional[np.ndarray]:
    """Return first frame of video, cached. Used for carousel peek."""
    if filepath in _video_first_frame_cache:
        return _video_first_frame_cache[filepath]
    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        _video_first_frame_cache[filepath] = frame
        return frame
    return None


def get_gallery_items(output_dir: Path) -> List[Tuple[Path, str, datetime]]:
    """
    Get all screenshots and videos from the output directory.
    Returns list of tuples: (filepath, type, modification_time)
    Sorted by modification time (newest first).
    """
    items = []

    if not output_dir.exists():
        return items

    for img_file in output_dir.glob("screenshot_*.png"):
        mtime = datetime.fromtimestamp(img_file.stat().st_mtime)
        items.append((img_file, "image", mtime))

    for vid_file in output_dir.glob("recording_*.mp4"):
        mtime = datetime.fromtimestamp(vid_file.stat().st_mtime)
        items.append((vid_file, "video", mtime))

    items.sort(key=lambda x: x[2], reverse=True)

    return items


def _update_viewer_swipe_inertia(frame_w: int) -> None:
    """Update viewer swipe offset from inertia; may advance to prev/next when crossing threshold."""
    if not button_state.gallery_viewer_swipe_inertia_active:
        return
    now = time.perf_counter()
    dt = max(1e-6, now - button_state.gallery_viewer_swipe_last_inertia_t)
    button_state.gallery_viewer_swipe_last_inertia_t = now
    vel = button_state.gallery_viewer_swipe_velocity
    new_offset = button_state.gallery_viewer_swipe_offset + vel * dt
    decay = max(0.0, 1.0 - ui_cache.VIEWER_SWIPE_FRICTION * dt)
    button_state.gallery_viewer_swipe_velocity = vel * decay
    if abs(button_state.gallery_viewer_swipe_velocity) < ui_cache.VIEWER_SWIPE_STOP_VELOCITY:
        button_state.gallery_viewer_swipe_velocity = 0.0
        button_state.gallery_viewer_swipe_inertia_active = False
    # Rubber band at ends: damp offset so it doesn't fly off
    n_items = getattr(button_state, "_gallery_items_len", 1)
    idx = button_state.gallery_selected_item or 0
    if idx <= 0 and new_offset < 0:
        new_offset *= VIEWER_RUBBER_BAND_FACTOR
    if idx >= n_items - 1 and new_offset > 0:
        new_offset *= VIEWER_RUBBER_BAND_FACTOR

    # Snap or advance: threshold is fraction of slot width so drag feels continuous
    slot_w = int(frame_w * VIEWER_CARD_WIDTH_RATIO) + int(frame_w * VIEWER_CARD_GAP_RATIO)
    threshold = max(float(ui_cache.VIEWER_SWIPE_THRESHOLD_PX), slot_w * VIEWER_SWIPE_SNAP_RATIO)
    if new_offset >= threshold:
        button_state.gallery_viewer_swipe_offset = 0.0
        button_state.gallery_viewer_swipe_velocity = 0.0
        button_state.gallery_viewer_swipe_inertia_active = False
        button_state.gallery_selected_item = min(
            (button_state.gallery_selected_item or 0) + 1,
            getattr(button_state, "_gallery_items_len", 1) - 1,
        )
        if button_state.gallery_viewer_mode == "video":
            button_state.gallery_video_playing = False
            button_state.gallery_video_frame_idx = 0
        return
    if new_offset <= -threshold:
        button_state.gallery_viewer_swipe_offset = 0.0
        button_state.gallery_viewer_swipe_velocity = 0.0
        button_state.gallery_viewer_swipe_inertia_active = False
        button_state.gallery_selected_item = max((button_state.gallery_selected_item or 0) - 1, 0)
        if button_state.gallery_viewer_mode == "video":
            button_state.gallery_video_playing = False
            button_state.gallery_video_frame_idx = 0
        return
    button_state.gallery_viewer_swipe_offset = new_offset


def draw_image_viewer(frame: np.ndarray, items: List[Tuple[Path, str, datetime]], output_dir: Optional[Path]) -> None:
    """Draw image viewer with shared dock and swipe/inertia."""
    frame[:] = (0, 0, 0)
    frame_h, frame_w = frame.shape[:2]

    if button_state.gallery_selected_item is None or button_state.gallery_selected_item >= len(items):
        return

    button_state._gallery_items_len = len(items)
    _update_viewer_swipe_inertia(frame_w)

    filepath, item_type, mtime = items[button_state.gallery_selected_item]
    font = cv2.FONT_HERSHEY_SIMPLEX
    controls_y = draw_viewer_chrome(frame, filepath, item_type, mtime, is_video=False)

    available_h = controls_y
    dx_float = _viewer_rubber_band_offset(
        button_state.gallery_viewer_swipe_offset,
        button_state.gallery_selected_item or 0,
        len(items),
    )
    idx = button_state.gallery_selected_item or 0
    n_items = len(items)
    slot_w = int(frame_w * VIEWER_CARD_WIDTH_RATIO) + int(frame_w * VIEWER_CARD_GAP_RATIO)
    button_state._gallery_swipe_threshold_px = max(50, int(slot_w * VIEWER_SWIPE_SNAP_RATIO))
    current_image_failed = False

    if _viewer_at_rest():
        # Only current card, full width, centered → black side borders
        path, itype, _ = items[idx]
        if itype == "image":
            img = cv2.imread(str(path))
            if img is not None:
                card = _make_image_card(img, frame_w, available_h)
                _blit_card_into_view(frame, card, 0, frame_w, frame_w, available_h)
            else:
                current_image_failed = True
        else:
            card = np.zeros((available_h, frame_w, 3), dtype=np.uint8)
            card[:] = (25, 25, 25)
            cv2.putText(card, "[VID]", (frame_w // 2 - 30, available_h // 2),
                       font, 0.8, (120, 120, 120), 1, cv2.LINE_AA)
            _blit_card_into_view(frame, card, 0, frame_w, frame_w, available_h)
    else:
        # Carousel: only valid indices; at edge swiping toward void draw only current (no ghost card)
        card_w = int(frame_w * VIEWER_CARD_WIDTH_RATIO)
        gap = int(frame_w * VIEWER_CARD_GAP_RATIO)
        center_offset = (frame_w - card_w) // 2
        draw_indices = _carousel_indices(idx, n_items)
        # At edge: during inertia never show neighbor (no ghost). When dragging, show neighbor only past threshold.
        edge_px = VIEWER_EDGE_PEEK_THRESHOLD_PX
        inertia_active = getattr(button_state, "gallery_viewer_swipe_inertia_active", False)
        if idx == 0:
            if inertia_active or dx_float >= -edge_px:
                draw_indices = [0]  # first: only current during inertia or until swipe left past threshold
        elif idx == n_items - 1:
            if inertia_active or dx_float <= edge_px:
                draw_indices = [n_items - 1]  # last: only current during inertia or until swipe right past threshold
        for i in draw_indices:
            if i < 0 or i >= n_items:
                continue
            path, itype, _ = items[i]
            if itype != "image":
                card = np.zeros((available_h, card_w, 3), dtype=np.uint8)
                card[:] = (30, 30, 30)
                cv2.putText(card, "[VID]", (card_w // 2 - 30, available_h // 2),
                           font, 0.8, (120, 120, 120), 1, cv2.LINE_AA)
            else:
                img = cv2.imread(str(path))
                if img is None:
                    current_image_failed = current_image_failed or (i == idx)
                    card = np.zeros((available_h, card_w, 3), dtype=np.uint8)
                    card[:] = (25, 25, 25)
                else:
                    card = _make_image_card(img, card_w, available_h)
            card_left = int(center_offset + (i - idx) * slot_w + dx_float)
            _blit_card_into_view(frame, card, card_left, card_w, frame_w, available_h)
        # Explicit black fill for void at edges so no content bleeds through
        if idx == n_items - 1:
            void_left = min(frame_w, int(center_offset + dx_float + card_w))
            if void_left < frame_w:
                frame[0:available_h, void_left:frame_w] = (0, 0, 0)
        elif idx == 0:
            void_right = max(0, int(center_offset + dx_float))
            if void_right > 0:
                frame[0:available_h, 0:void_right] = (0, 0, 0)

    if current_image_failed:
        msg = "Failed to load image"
        (msg_w, msg_h), _ = cv2.getTextSize(msg, font, 0.7, 1)
        cv2.putText(frame, msg, ((frame_w - msg_w) // 2, frame_h // 2),
                   font, 0.7, (150, 150, 150), 1, cv2.LINE_AA)

    # Prev/next arrows (both flipped 180° from default: next "<", prev ">")
    arrow_y = controls_y // 2 - 28
    next_btn_x, next_btn_w, next_btn_h = 20, 50, 56
    prev_btn_x, prev_btn_w, prev_btn_h = frame_w - 70, 50, 56
    if button_state.gallery_selected_item < len(items) - 1:
        if "gallery_next" not in menu_buttons:
            menu_buttons["gallery_next"] = Button(next_btn_x, arrow_y, next_btn_w, next_btn_h, "<")
        else:
            menu_buttons["gallery_next"].x, menu_buttons["gallery_next"].y = next_btn_x, arrow_y
            menu_buttons["gallery_next"].w, menu_buttons["gallery_next"].h = next_btn_w, next_btn_h
        menu_buttons["gallery_next"].draw(frame, transparent=True)
        draw_viewer_button_feedback(frame, "gallery_next", next_btn_x, arrow_y, next_btn_w, next_btn_h)
    if button_state.gallery_selected_item > 0:
        if "gallery_prev" not in menu_buttons:
            menu_buttons["gallery_prev"] = Button(prev_btn_x, arrow_y, prev_btn_w, prev_btn_h, ">")
        else:
            menu_buttons["gallery_prev"].x, menu_buttons["gallery_prev"].y = prev_btn_x, arrow_y
            menu_buttons["gallery_prev"].w, menu_buttons["gallery_prev"].h = prev_btn_w, prev_btn_h
        menu_buttons["gallery_prev"].draw(frame, transparent=True)
        draw_viewer_button_feedback(frame, "gallery_prev", prev_btn_x, arrow_y, prev_btn_w, prev_btn_h)

    draw_viewer_back_button_on_top(frame)


def draw_video_viewer(frame: np.ndarray, items: List[Tuple[Path, str, datetime]], output_dir: Optional[Path]) -> None:
    """Draw video player with shared dock and swipe/inertia."""
    frame[:] = (0, 0, 0)
    frame_h, frame_w = frame.shape[:2]

    if button_state.gallery_selected_item is None or button_state.gallery_selected_item >= len(items):
        return

    button_state._gallery_items_len = len(items)
    _update_viewer_swipe_inertia(frame_w)

    filepath, item_type, mtime = items[button_state.gallery_selected_item]
    cap = cv2.VideoCapture(str(filepath))

    if not cap.isOpened():
        font = cv2.FONT_HERSHEY_SIMPLEX
        msg = "Failed to load video"
        (msg_w, msg_h), _ = cv2.getTextSize(msg, font, 0.7, 1)
        cv2.putText(frame, msg, ((frame_w - msg_w) // 2, frame_h // 2),
                   font, 0.7, (150, 150, 150), 1, cv2.LINE_AA)
        controls_y = draw_viewer_chrome(frame, filepath, item_type, mtime, is_video=True,
                                        total_frames=0, fps=0, current_idx=0, play_text="PLAY")
        draw_viewer_back_button_on_top(frame)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    button_state._gallery_video_total_frames = total_frames
    button_state._gallery_video_fps = fps

    if button_state.gallery_video_playing:
        button_state.gallery_video_frame_idx += 1
        if button_state.gallery_video_frame_idx >= total_frames:
            button_state.gallery_video_frame_idx = 0

    button_state.gallery_video_frame_idx = max(0, min(button_state.gallery_video_frame_idx, total_frames - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, button_state.gallery_video_frame_idx)
    ret, vid_frame = cap.read()
    cap.release()

    play_text = "PAUSE" if button_state.gallery_video_playing else "PLAY"
    controls_y = draw_viewer_chrome(frame, filepath, item_type, mtime, is_video=True,
                                   total_frames=total_frames, fps=fps,
                                   current_idx=button_state.gallery_video_frame_idx, play_text=play_text)

    available_h = controls_y
    dx_float = _viewer_rubber_band_offset(
        button_state.gallery_viewer_swipe_offset,
        button_state.gallery_selected_item or 0,
        len(items),
    )
    idx = button_state.gallery_selected_item or 0
    n_items = len(items)
    slot_w = int(frame_w * VIEWER_CARD_WIDTH_RATIO) + int(frame_w * VIEWER_CARD_GAP_RATIO)
    button_state._gallery_swipe_threshold_px = max(50, int(slot_w * VIEWER_SWIPE_SNAP_RATIO))
    font = cv2.FONT_HERSHEY_SIMPLEX

    if _viewer_at_rest():
        # Only current card, full width, black borders
        if ret and vid_frame is not None:
            card = _make_image_card(vid_frame, frame_w, available_h)
            _blit_card_into_view(frame, card, 0, frame_w, frame_w, available_h)
        else:
            card = np.zeros((available_h, frame_w, 3), dtype=np.uint8)
            card[:] = (25, 25, 25)
            _blit_card_into_view(frame, card, 0, frame_w, frame_w, available_h)
    else:
        card_w = int(frame_w * VIEWER_CARD_WIDTH_RATIO)
        gap = int(frame_w * VIEWER_CARD_GAP_RATIO)
        center_offset = (frame_w - card_w) // 2
        draw_indices = _carousel_indices(idx, n_items)
        edge_px = VIEWER_EDGE_PEEK_THRESHOLD_PX
        inertia_active = getattr(button_state, "gallery_viewer_swipe_inertia_active", False)
        if idx == 0 and (inertia_active or dx_float >= -edge_px):
            draw_indices = [0]
        elif idx == n_items - 1 and (inertia_active or dx_float <= edge_px):
            draw_indices = [n_items - 1]
        for i in draw_indices:
            if i < 0 or i >= n_items:
                continue
            path, itype, _ = items[i]
            if i == idx and ret and vid_frame is not None:
                card = _make_image_card(vid_frame, card_w, available_h)
            elif itype == "video":
                first = _get_video_first_frame(path)
                if first is not None:
                    card = _make_image_card(first, card_w, available_h)
                else:
                    card = np.zeros((available_h, card_w, 3), dtype=np.uint8)
                    card[:] = (30, 30, 30)
                    cv2.putText(card, "[VID]", (card_w // 2 - 30, available_h // 2),
                               font, 0.8, (120, 120, 120), 1, cv2.LINE_AA)
            else:
                img = cv2.imread(str(path))
                if img is not None:
                    card = _make_image_card(img, card_w, available_h)
                else:
                    card = np.zeros((available_h, card_w, 3), dtype=np.uint8)
                    card[:] = (25, 25, 25)
            card_left = int(center_offset + (i - idx) * slot_w + dx_float)
            _blit_card_into_view(frame, card, card_left, card_w, frame_w, available_h)
        if idx == n_items - 1:
            void_left = min(frame_w, int(center_offset + dx_float + card_w))
            if void_left < frame_w:
                frame[0:available_h, void_left:frame_w] = (0, 0, 0)
        elif idx == 0:
            void_right = max(0, int(center_offset + dx_float))
            if void_right > 0:
                frame[0:available_h, 0:void_right] = (0, 0, 0)

    arrow_y = controls_y // 2 - 28
    next_btn_x, next_btn_w, next_btn_h = 20, 50, 56
    prev_btn_x, prev_btn_w, prev_btn_h = frame_w - 70, 50, 56
    if button_state.gallery_selected_item < len(items) - 1:
        if "gallery_next" not in menu_buttons:
            menu_buttons["gallery_next"] = Button(next_btn_x, arrow_y, next_btn_w, next_btn_h, "<")
        else:
            menu_buttons["gallery_next"].x, menu_buttons["gallery_next"].y = next_btn_x, arrow_y
            menu_buttons["gallery_next"].w, menu_buttons["gallery_next"].h = next_btn_w, next_btn_h
        menu_buttons["gallery_next"].draw(frame, transparent=True)
        draw_viewer_button_feedback(frame, "gallery_next", next_btn_x, arrow_y, next_btn_w, next_btn_h)
    if button_state.gallery_selected_item > 0:
        if "gallery_prev" not in menu_buttons:
            menu_buttons["gallery_prev"] = Button(prev_btn_x, arrow_y, prev_btn_w, prev_btn_h, ">")
        else:
            menu_buttons["gallery_prev"].x, menu_buttons["gallery_prev"].y = prev_btn_x, arrow_y
            menu_buttons["gallery_prev"].w, menu_buttons["gallery_prev"].h = prev_btn_w, prev_btn_h
        menu_buttons["gallery_prev"].draw(frame, transparent=True)
        draw_viewer_button_feedback(frame, "gallery_prev", prev_btn_x, arrow_y, prev_btn_w, prev_btn_h)

    draw_viewer_back_button_on_top(frame)


def draw_delete_modal(
    frame: np.ndarray,
    title: str = "Delete this item?",
    subtitle: str = "This action cannot be undone.",
    yes_text: str = "YES, DELETE",
    no_text: str = "CANCEL",
) -> None:
    modal_w = 400
    modal_h = 180
    modal_x = (frame.shape[1] - modal_w) // 2
    modal_y = (frame.shape[0] - modal_h) // 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), (100, 100, 100), 3, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX

    title_scale = 0.7
    (title_w, title_h), _ = cv2.getTextSize(title, font, title_scale, 2)
    title_x = modal_x + (modal_w - title_w) // 2
    title_y = modal_y + 50
    cv2.putText(frame, title, (title_x, title_y), font, title_scale, (255, 255, 255), 2, cv2.LINE_AA)

    msg_scale = 0.5
    (msg_w, msg_h), _ = cv2.getTextSize(subtitle, font, msg_scale, 1)
    msg_x = modal_x + (modal_w - msg_w) // 2
    msg_y = modal_y + 85
    cv2.putText(frame, subtitle, (msg_x, msg_y), font, msg_scale, (200, 200, 200), 1, cv2.LINE_AA)

    yes_btn_w = 160
    yes_btn_h = 45
    yes_btn_x = modal_x + (modal_w // 2) - yes_btn_w - 15
    yes_btn_y = modal_y + modal_h - yes_btn_h - 25

    if "modal_yes" not in menu_buttons:
        menu_buttons["modal_yes"] = Button(yes_btn_x, yes_btn_y, yes_btn_w, yes_btn_h, yes_text)
    else:
        b = menu_buttons["modal_yes"]
        b.x, b.y, b.w, b.h, b.text = yes_btn_x, yes_btn_y, yes_btn_w, yes_btn_h, yes_text

    menu_buttons["modal_yes"].is_active = True
    menu_buttons["modal_yes"].draw(frame, transparent=True, active_color=(0, 0, 220))

    no_btn_w = 140
    no_btn_h = 45
    no_btn_x = modal_x + (modal_w // 2) + 15
    no_btn_y = modal_y + modal_h - no_btn_h - 25

    if "modal_no" not in menu_buttons:
        menu_buttons["modal_no"] = Button(no_btn_x, no_btn_y, no_btn_w, no_btn_h, no_text)
    else:
        b = menu_buttons["modal_no"]
        b.x, b.y, b.w, b.h, b.text = no_btn_x, no_btn_y, no_btn_w, no_btn_h, no_text

    menu_buttons["modal_no"].draw(frame, transparent=True)


def draw_gallery_view(frame: np.ndarray, output_dir: Optional[Path]) -> None:
    """
    Draw the gallery view showing all saved screenshots and videos.
    Displays items in a grid layout with thumbnails.
    Supports scrolling and clickable thumbnails.
    """
    if output_dir is None:
        return

    items = get_gallery_items(output_dir)

    if button_state.gallery_viewer_mode == "image":
        draw_image_viewer(frame, items, output_dir)
        if button_state.gallery_delete_modal_open:
            draw_delete_modal(frame)
        return
    elif button_state.gallery_viewer_mode == "video":
        draw_video_viewer(frame, items, output_dir)
        if button_state.gallery_delete_modal_open:
            draw_delete_modal(frame)
        return

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)

    header_h = 80
    cv2.rectangle(frame, (0, 0), (frame.shape[1], header_h), (25, 25, 25), -1)
    cv2.line(frame, (0, header_h), (frame.shape[1], header_h), (80, 80, 80), 2)

    title = "GALLERY"
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 1.2
    title_thick = 2
    (title_w, title_h), _ = cv2.getTextSize(title, font, title_scale, title_thick)
    title_x = (frame.shape[1] - title_w) // 2
    title_y = (header_h + title_h) // 2 + 5 - 12
    cv2.putText(frame, title, (title_x, title_y), font, title_scale, (255, 255, 255), title_thick, cv2.LINE_8)
    
    if button_state.gallery_select_mode:
        selected_count = len(button_state.gallery_selected_items)
        if selected_count > 0:
            info_text = f"Total: {len(items)} items | Selected: {selected_count} | Click to select/deselect | Swipe to scroll"
        else:
            info_text = f"Total: {len(items)} items | Click to select/deselect | Swipe to scroll"
    else:
        info_text = f"Total: {len(items)} items ({sum(1 for _, t, _ in items if t == 'image')} images, {sum(1 for _, t, _ in items if t == 'video')} videos) | Click to view | Swipe to scroll"
    
    info_scale = 0.45
    (info_w, info_h), _ = cv2.getTextSize(info_text, font, info_scale, 1)
    info_x = (frame.shape[1] - info_w) // 2
    info_y = title_y + 22
    cv2.putText(frame, info_text, (info_x, info_y), font, info_scale, (150, 150, 150), 1, cv2.LINE_AA)

    back_btn_x = 10
    back_btn_y = 20
    back_btn_w = 75
    back_btn_h = 40

    if "gallery_back" not in menu_buttons:
        menu_buttons["gallery_back"] = Button(back_btn_x, back_btn_y, back_btn_w, back_btn_h, "")
    else:
        menu_buttons["gallery_back"].x = back_btn_x
        menu_buttons["gallery_back"].y = back_btn_y
        menu_buttons["gallery_back"].w = back_btn_w
        menu_buttons["gallery_back"].h = back_btn_h

    menu_buttons["gallery_back"].draw(frame, transparent=True, icon_type="back")
    
    if button_state.gallery_select_mode and button_state.gallery_selected_items:
        selected_count = len(button_state.gallery_selected_items)
        delete_btn_w = 140
        delete_btn_x = back_btn_x + back_btn_w + 10
        delete_btn_y = 20
        
        delete_text = f"DELETE ({selected_count})"
        delete_color = (0, 0, 220)
        
        if "gallery_delete_selected" not in menu_buttons:
            menu_buttons["gallery_delete_selected"] = Button(delete_btn_x, delete_btn_y, delete_btn_w, back_btn_h, delete_text)
        else:
            menu_buttons["gallery_delete_selected"].x = delete_btn_x
            menu_buttons["gallery_delete_selected"].y = delete_btn_y
            menu_buttons["gallery_delete_selected"].w = delete_btn_w
            menu_buttons["gallery_delete_selected"].h = back_btn_h
            menu_buttons["gallery_delete_selected"].text = delete_text
        
        menu_buttons["gallery_delete_selected"].is_active = True
        menu_buttons["gallery_delete_selected"].draw(frame, transparent=True, active_color=delete_color)

    if items:
        btn_gap = 10
        btn_h = 40

        right_margin = 20
        current_x = frame.shape[1] - right_margin

        select_mode_btn_w = 100
        select_mode_btn_x = current_x - select_mode_btn_w
        select_mode_btn_y = 20

        if "gallery_select_mode" not in menu_buttons:
            menu_buttons["gallery_select_mode"] = Button(select_mode_btn_x, select_mode_btn_y, select_mode_btn_w, btn_h, "SELECT")
        else:
            menu_buttons["gallery_select_mode"].x = select_mode_btn_x
            menu_buttons["gallery_select_mode"].y = select_mode_btn_y
            menu_buttons["gallery_select_mode"].w = select_mode_btn_w
            menu_buttons["gallery_select_mode"].h = btn_h
            menu_buttons["gallery_select_mode"].text = "DONE" if button_state.gallery_select_mode else "SELECT"

        menu_buttons["gallery_select_mode"].is_active = button_state.gallery_select_mode

        if button_state.gallery_select_mode:
            menu_buttons["gallery_select_mode"].draw(frame, transparent=True, active_color=(200, 100, 40))
        else:
            menu_buttons["gallery_select_mode"].draw(frame, transparent=True)

        current_x = select_mode_btn_x - btn_gap

        if button_state.gallery_select_mode:
            select_all_btn_w = 120
            select_all_btn_x = current_x - select_all_btn_w
            select_all_btn_y = 20

            if "gallery_select_all" not in menu_buttons:
                menu_buttons["gallery_select_all"] = Button(select_all_btn_x, select_all_btn_y, select_all_btn_w, btn_h, "SELECT ALL")
            else:
                menu_buttons["gallery_select_all"].x = select_all_btn_x
                menu_buttons["gallery_select_all"].y = select_all_btn_y
                menu_buttons["gallery_select_all"].w = select_all_btn_w
                menu_buttons["gallery_select_all"].h = btn_h
                all_selected = len(button_state.gallery_selected_items) == len(items)
                menu_buttons["gallery_select_all"].text = "DESELECT ALL" if all_selected else "SELECT ALL"

            menu_buttons["gallery_select_all"].draw(frame, transparent=True)

    if items:
        total_media_size = sum(filepath.stat().st_size for filepath, _, _ in items)
        
        if output_dir and output_dir.exists():
            try:
                disk_usage = shutil.disk_usage(str(output_dir))
                total_space = disk_usage.total
                used_space = disk_usage.used
                free_space = disk_usage.free
            except:
                total_space = 128 * 1024 * 1024 * 1024
                used_space = total_media_size
                free_space = total_space - used_space
        else:
            total_space = 128 * 1024 * 1024 * 1024
            used_space = total_media_size
            free_space = total_space - used_space
        
        def format_size(size_bytes):
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size_bytes < 1024.0:
                    return f"{size_bytes:.1f} {unit}"
                size_bytes /= 1024.0
            return f"{size_bytes:.1f} PB"
        
        bar_w = 25
        bar_h = 300
        bar_x = frame.shape[1] - 50
        bar_y = header_h + 80
        
        label_text = "STORAGE"
        label_scale = 0.4
        (label_w, label_h), _ = cv2.getTextSize(label_text, font, label_scale, 1)
        label_x = bar_x + (bar_w - label_w) // 2
        label_y = bar_y - 10
        cv2.putText(frame, label_text, (label_x, label_y), font, label_scale, (200, 200, 200), 1, cv2.LINE_AA)
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        
        disk_usage_percent = (used_space / total_space * 100) if total_space > 0 else 0
        used_h = int(bar_h * min(disk_usage_percent / 100.0, 1.0))
        
        if used_h > 0:
            used_color = (80, 200, 80) if disk_usage_percent < 75 else (80, 180, 220) if disk_usage_percent < 90 else (80, 80, 220)
            used_y = bar_y + bar_h - used_h
            cv2.rectangle(frame, (bar_x, used_y), (bar_x + bar_w, bar_y + bar_h), used_color, -1)
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 2, cv2.LINE_AA)
        
        if disk_usage_percent < 0.1:
            percent_text = f"{disk_usage_percent:.2f}%"
        else:
            percent_text = f"{disk_usage_percent:.1f}%"
        percent_scale = 0.35
        (percent_w, percent_h), _ = cv2.getTextSize(percent_text, font, percent_scale, 1)
        percent_x = bar_x + (bar_w - percent_w) // 2
        percent_y = bar_y + bar_h + 15
        cv2.putText(frame, percent_text, (percent_x, percent_y), font, percent_scale, (200, 200, 200), 1, cv2.LINE_AA)
        
        used_text = format_size(used_space)
        used_scale = 0.32
        (used_w, used_h), _ = cv2.getTextSize(used_text, font, used_scale, 1)
        used_x = bar_x + (bar_w - used_w) // 2
        used_y = percent_y + 15
        cv2.putText(frame, used_text, (used_x, used_y), font, used_scale, (180, 180, 180), 1, cv2.LINE_AA)
        
        legend_box_size = 8
        legend_box_x = bar_x + (bar_w - legend_box_size) // 2
        legend_box_y = used_y + 14
        legend_color = (80, 200, 80) if disk_usage_percent < 75 else (80, 180, 220) if disk_usage_percent < 90 else (80, 80, 220)
        cv2.rectangle(frame, (legend_box_x, legend_box_y), (legend_box_x + legend_box_size, legend_box_y + legend_box_size), legend_color, -1)
        cv2.rectangle(frame, (legend_box_x, legend_box_y), (legend_box_x + legend_box_size, legend_box_y + legend_box_size), (120, 120, 120), 1, cv2.LINE_AA)
        
        legend_text = "Used"
        legend_scale = 0.28
        (legend_w, legend_h), _ = cv2.getTextSize(legend_text, font, legend_scale, 1)
        legend_text_x = bar_x + (bar_w - legend_w) // 2
        legend_text_y = legend_box_y + legend_box_size + 10
        cv2.putText(frame, legend_text, (legend_text_x, legend_text_y), font, legend_scale, (150, 150, 150), 1, cv2.LINE_AA)
        
        total_text = format_size(total_space)
        total_scale = 0.3
        (total_w, total_h), _ = cv2.getTextSize(total_text, font, total_scale, 1)
        total_x = bar_x + (bar_w - total_w) // 2
        total_y = legend_text_y + 13
        cv2.putText(frame, total_text, (total_x, total_y), font, total_scale, (150, 150, 150), 1, cv2.LINE_AA)

    if not items:
        msg = "No captures yet. Use SHOT or REC to create content."
        msg_scale = 0.7
        msg_thick = 1
        (msg_w, msg_h), _ = cv2.getTextSize(msg, font, msg_scale, msg_thick)
        msg_x = (frame.shape[1] - msg_w) // 2
        msg_y = frame.shape[0] // 2
        cv2.putText(frame, msg, (msg_x, msg_y), font, msg_scale, (150, 150, 150), msg_thick, cv2.LINE_AA)
        return

    margin = 20
    grid_start_y = header_h + margin
    grid_start_x = margin

    thumb_w = 280
    thumb_h = 180
    gap = 15

    cols = (frame.shape[1] - 2 * margin + gap) // (thumb_w + gap)
    cols = max(1, cols)

    total_rows = (len(items) + cols - 1) // cols
    item_height = thumb_h + gap + 50
    total_content_height = grid_start_y + (total_rows * item_height)
    footer_space = 40

    visible_height = frame.shape[0] - header_h - footer_space
    max_scroll = max(0, total_content_height - frame.shape[0] + footer_space)

    if button_state.gallery_inertia_active and not button_state.gallery_dragging:
        now = time.perf_counter()
        dt = max(1e-6, now - getattr(button_state, "gallery_last_inertia_t", now))
        button_state.gallery_last_inertia_t = now

        new_scroll = float(button_state.gallery_scroll_offset) + button_state.gallery_scroll_velocity * dt

        FRICTION = 1.2
        decay = max(0.0, 1.0 - FRICTION * dt)
        button_state.gallery_scroll_velocity *= decay

        max_scroll_f = float(getattr(button_state, "gallery_max_scroll", 0))
        if new_scroll < 0.0:
            new_scroll = 0.0
            button_state.gallery_scroll_velocity = 0.0
            button_state.gallery_inertia_active = False
        elif new_scroll > max_scroll_f:
            new_scroll = max_scroll_f
            button_state.gallery_scroll_velocity = 0.0
            button_state.gallery_inertia_active = False

        if abs(button_state.gallery_scroll_velocity) < 15.0:
            button_state.gallery_scroll_velocity = 0.0
            button_state.gallery_inertia_active = False

        button_state.gallery_scroll_offset = int(new_scroll)

    button_state.gallery_max_scroll = int(max_scroll)
    button_state.gallery_scroll_offset = max(0, min(button_state.gallery_scroll_offset, max_scroll))

    clip_y_start = header_h

    if not hasattr(button_state, 'gallery_thumbnail_rects'):
        button_state.gallery_thumbnail_rects = []
    button_state.gallery_thumbnail_rects.clear()

    scroll_offset = button_state.gallery_scroll_offset

    for idx, (filepath, item_type, mtime) in enumerate(items):
        row = idx // cols
        col = idx % cols

        x = grid_start_x + col * (thumb_w + gap)
        y = grid_start_y + row * (thumb_h + gap + 50) - scroll_offset

        if y + thumb_h + 50 < header_h or y > frame.shape[0]:
            continue

        button_state.gallery_thumbnail_rects.append({
            'idx': idx,
            'x': x,
            'y': y,
            'w': thumb_w,
            'h': thumb_h,
            'filepath': filepath,
            'type': item_type
        })

        vis_top = header_h
        vis_bot = frame.shape[0]

        x0, y0 = x, y
        x1, y1 = x + thumb_w, y + thumb_h

        cx0 = max(x0, 0)
        cx1 = min(x1, frame.shape[1])
        cy0 = max(y0, vis_top)
        cy1 = min(y1, vis_bot)

        if cx1 <= cx0 or cy1 <= cy0:
            continue

        is_selected = button_state.gallery_select_mode and idx in button_state.gallery_selected_items

        cv2.rectangle(frame, (cx0, cy0), (cx1, cy1), (40, 40, 40), -1)

        fully_visible = (y0 >= vis_top) and (y1 <= vis_bot)
        if fully_visible:
            border_color = (80, 255, 100) if is_selected else (100, 100, 100)
            border_thickness = 4 if is_selected else 2
            cv2.rectangle(frame, (x0, y0), (x1, y1), border_color, border_thickness, cv2.LINE_AA)

        if is_selected:
            check_size = 25
            check_x = x + thumb_w - check_size - 10
            check_y = y + 10

            cv2.circle(frame, (check_x + check_size//2, check_y + check_size//2), check_size//2, (80, 255, 100), -1, cv2.LINE_AA)

            check_pts = np.array([
                [check_x + 6, check_y + check_size//2],
                [check_x + check_size//2 - 2, check_y + check_size - 8],
                [check_x + check_size - 6, check_y + 5]
            ], np.int32)
            cv2.polylines(frame, [check_pts], False, (255, 255, 255), 3, cv2.LINE_AA)

        try:
            if item_type == "image":
                mtime_s = filepath.stat().st_mtime
                cached = ui_cache._THUMB_CACHE.get(filepath)
                cached_m = ui_cache._THUMB_CACHE_MTIME.get(filepath)

                if cached is None or cached_m != mtime_s:
                    img = cv2.imread(str(filepath))
                    if img is not None:
                        cached = cv2.resize(img, (thumb_w - 4, thumb_h - 4), interpolation=cv2.INTER_AREA)
                        ui_cache._THUMB_CACHE[filepath] = cached
                        ui_cache._THUMB_CACHE_MTIME[filepath] = mtime_s

                if cached is not None:
                    dst_x0 = x + 2
                    dst_y0 = y + 2
                    dst_x1 = x + thumb_w - 2
                    dst_y1 = y + thumb_h - 2

                    cdx0 = max(dst_x0, 0)
                    cdx1 = min(dst_x1, frame.shape[1])
                    cdy0 = max(dst_y0, vis_top)
                    cdy1 = min(dst_y1, vis_bot)

                    if cdx1 > cdx0 and cdy1 > cdy0:
                        sx0 = cdx0 - dst_x0
                        sy0 = cdy0 - dst_y0
                        sx1 = sx0 + (cdx1 - cdx0)
                        sy1 = sy0 + (cdy1 - cdy0)

                        frame[cdy0:cdy1, cdx0:cdx1] = cached[sy0:sy1, sx0:sx1]
            elif item_type == "video":
                mtime_s = filepath.stat().st_mtime
                cached = ui_cache._THUMB_CACHE.get(filepath)
                cached_m = ui_cache._THUMB_CACHE_MTIME.get(filepath)

                if cached is None or cached_m != mtime_s:
                    cap = cv2.VideoCapture(str(filepath))
                    ret, vid_frame = cap.read()
                    cap.release()

                    if ret and vid_frame is not None:
                        cached = cv2.resize(vid_frame, (thumb_w - 4, thumb_h - 4), interpolation=cv2.INTER_AREA)
                        ui_cache._THUMB_CACHE[filepath] = cached
                        ui_cache._THUMB_CACHE_MTIME[filepath] = mtime_s

                if cached is not None:
                    dst_x0 = x + 2
                    dst_y0 = y + 2
                    dst_x1 = x + thumb_w - 2
                    dst_y1 = y + thumb_h - 2

                    cdx0 = max(dst_x0, 0)
                    cdx1 = min(dst_x1, frame.shape[1])
                    cdy0 = max(dst_y0, vis_top)
                    cdy1 = min(dst_y1, vis_bot)

                    if cdx1 > cdx0 and cdy1 > cdy0:
                        sx0 = cdx0 - dst_x0
                        sy0 = cdy0 - dst_y0
                        sx1 = sx0 + (cdx1 - cdx0)
                        sy1 = sy0 + (cdy1 - cdy0)

                        frame[cdy0:cdy1, cdx0:cdx1] = cached[sy0:sy1, sx0:sx1]

                    center_x = x + thumb_w // 2
                    center_y = y + thumb_h // 2
                    play_size = 30

                    ox0 = center_x - play_size
                    oy0 = center_y - play_size
                    ox1 = center_x + play_size
                    oy1 = center_y + play_size

                    cox0 = max(ox0, 0)
                    cox1 = min(ox1, frame.shape[1])
                    coy0 = max(oy0, vis_top)
                    coy1 = min(oy1, vis_bot)

                    if cox1 > cox0 and coy1 > coy0:
                        roi = frame[coy0:coy1, cox0:cox1]
                        overlay_roi = roi.copy()

                        ccx = center_x - cox0
                        ccy = center_y - coy0

                        cv2.circle(overlay_roi, (ccx, ccy), play_size, (0, 0, 0), -1, cv2.LINE_AA)
                        cv2.addWeighted(overlay_roi, 0.6, roi, 0.4, 0.0, dst=roi)

                        pts = np.array([
                            [center_x - 10, center_y - 15],
                            [center_x - 10, center_y + 15],
                            [center_x + 15, center_y]
                        ], np.int32)

                        if (pts[:, 0].min() >= 0 and pts[:, 0].max() < frame.shape[1] and
                            pts[:, 1].min() >= vis_top and pts[:, 1].max() < vis_bot):
                            cv2.fillPoly(frame, [pts], (255, 255, 255), cv2.LINE_AA)
        except Exception:
            cv2.putText(frame, "Error", (x + 10, y + thumb_h // 2),
                       font, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

        label_y = y + thumb_h + 20
        if label_y >= header_h and label_y <= frame.shape[0]:
            filename = filepath.name

            if len(filename) > 30:
                filename = filename[:27] + "..."

            type_icon = "[IMG]" if item_type == "image" else "[VID]"
            type_color = (100, 200, 100) if item_type == "image" else (100, 150, 255)

            cv2.putText(frame, type_icon, (x, label_y), font, 0.45, type_color, 1, cv2.LINE_AA)

            cv2.putText(frame, filename, (x, label_y + 20), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    if button_state.gallery_delete_modal_open:
        if button_state.gallery_delete_modal_kind == "batch":
            n = len(button_state.gallery_selected_items)
            draw_delete_modal(
                frame,
                title=f"Delete {n} item(s)?",
                subtitle="This action cannot be undone.",
                yes_text="YES, DELETE",
                no_text="CANCEL",
            )
        else:
            draw_delete_modal(frame)
