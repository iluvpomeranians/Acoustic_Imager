"""
Click and mouse handlers for menu, buttons, and gallery.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import time

from . import ui_cache
from .button import buttons, menu_buttons
from .gallery import get_displayed_gallery_items, _viewer_rubber_band_offset
from .viewer_dock import trigger_viewer_button_feedback
from .menu import get_recording_timestamp_rect
from .screenshot import save_screenshot
from ..config import SOURCE_MODES, SOURCE_DEFAULT
from ..state import button_state
from .video_recorder import VideoRecorder


def handle_gallery_click(x: int, y: int, output_dir: Optional[Path]) -> bool:
    """
    Handle clicks in gallery view.
    Returns True if click was handled in gallery, False otherwise.
    """
    if not button_state.gallery_open:
        return False

    if "gallery_back" in menu_buttons and menu_buttons["gallery_back"].contains(x, y):
        trigger_viewer_button_feedback("gallery_back")
        if button_state.gallery_viewer_mode == "grid":
            button_state.gallery_open = False
            button_state.gallery_scroll_offset = 0
            button_state.gallery_selected_item = None
            button_state.gallery_select_mode = False
            button_state.gallery_selected_items.clear()

        else:
            button_state.gallery_viewer_mode = "grid"
            button_state.gallery_video_playing = False
            button_state.gallery_video_frame_idx = 0

        return True

    if button_state.gallery_delete_modal_open:
        if "modal_yes" in menu_buttons and menu_buttons["modal_yes"].contains(x, y):
            items = get_displayed_gallery_items(output_dir)

            try:
                if button_state.gallery_delete_modal_kind == "batch":
                    deleted = 0
                    for idx in sorted(button_state.gallery_selected_items, reverse=True):
                        if 0 <= idx < len(items):
                            path = items[idx][0]
                            try:
                                path.unlink()
                                deleted += 1
                            except Exception as e:
                                print(f"Failed to delete {path}: {e}")

                    button_state.gallery_selected_items.clear()
                    button_state.gallery_select_mode = False
                    button_state.gallery_delete_modal_open = False
                    button_state.gallery_delete_modal_kind = "single"
                    if deleted > 0:
                        button_state.gallery_storage_dirty = True
                    return True

                if button_state.gallery_selected_item is not None and button_state.gallery_selected_item < len(items):
                    filepath = items[button_state.gallery_selected_item][0]
                    filepath.unlink()
                    button_state.gallery_storage_dirty = True

                    if len(items) <= 1:
                        button_state.gallery_viewer_mode = "grid"
                        button_state.gallery_selected_item = None
                    else:
                        if button_state.gallery_selected_item >= len(items) - 1:
                            button_state.gallery_selected_item = len(items) - 2

                button_state.gallery_delete_modal_open = False
                button_state.gallery_delete_modal_kind = "single"
            except Exception:
                button_state.gallery_delete_modal_open = False
                button_state.gallery_delete_modal_kind = "single"
            return True

        if "modal_no" in menu_buttons and menu_buttons["modal_no"].contains(x, y):
            button_state.gallery_delete_modal_open = False
            button_state.gallery_delete_modal_kind = "single"
            return True

        return True

    # Dock row buttons have priority; Search, Filter, Sort are mutually exclusive (one open at a time)
    if button_state.gallery_viewer_mode == "grid":
        if "gallery_dock_search" in menu_buttons and menu_buttons["gallery_dock_search"].contains(x, y):
            button_state.gallery_search_keyboard_open = not button_state.gallery_search_keyboard_open
            if button_state.gallery_search_keyboard_open:
                button_state.gallery_filter_modal_open = False
                button_state.gallery_sort_modal_open = False
            return True
        if "gallery_dock_filter" in menu_buttons and menu_buttons["gallery_dock_filter"].contains(x, y):
            button_state.gallery_filter_modal_open = not button_state.gallery_filter_modal_open
            if button_state.gallery_filter_modal_open:
                button_state.gallery_sort_modal_open = False
                button_state.gallery_search_keyboard_open = False
            return True
        if "gallery_dock_sort" in menu_buttons and menu_buttons["gallery_dock_sort"].contains(x, y):
            button_state.gallery_sort_modal_open = not button_state.gallery_sort_modal_open
            if button_state.gallery_sort_modal_open:
                button_state.gallery_filter_modal_open = False
                button_state.gallery_search_keyboard_open = False
            return True

    # Filter modal: option click only sets filter (modal stays open); click outside closes and fall through
    if button_state.gallery_filter_modal_open:
        for value in ("all", "image", "video"):
            key = f"gallery_filter_opt_{value}"
            if key in menu_buttons and menu_buttons[key].contains(x, y):
                button_state.gallery_filter_type = value
                return True
        if "gallery_filter_modal_panel" in menu_buttons and menu_buttons["gallery_filter_modal_panel"].contains(x, y):
            return True
        button_state.gallery_filter_modal_open = False
        # fall through so the same click can hit thumbnails or other controls

    # Sort modal: option click only sets sort (modal stays open); click outside closes and fall through
    if button_state.gallery_sort_modal_open:
        for value in ("date", "name", "size"):
            key = f"gallery_sort_opt_{value}"
            if key in menu_buttons and menu_buttons[key].contains(x, y):
                button_state.gallery_sort_by = value
                return True
        if "gallery_sort_modal_panel" in menu_buttons and menu_buttons["gallery_sort_modal_panel"].contains(x, y):
            return True
        button_state.gallery_sort_modal_open = False
        # fall through so the same click can hit thumbnails or other controls

    if button_state.gallery_search_keyboard_open:
        if "search_key_done" in menu_buttons and menu_buttons["search_key_done"].contains(x, y):
            button_state.gallery_search_keyboard_open = False
            return True
        if "search_key_clear" in menu_buttons and menu_buttons["search_key_clear"].contains(x, y):
            button_state.gallery_search_query = ""
            return True
        if "search_key_backspace" in menu_buttons and menu_buttons["search_key_backspace"].contains(x, y):
            button_state.gallery_search_query = (button_state.gallery_search_query or "")[:-1]
            return True
        for c in "abcdefghijklmnopqrstuvwxyz0123456789":
            key = f"search_key_{c}"
            if key in menu_buttons and menu_buttons[key].contains(x, y):
                button_state.gallery_search_query = (button_state.gallery_search_query or "") + c
                return True
        if "search_keyboard_panel" in menu_buttons and menu_buttons["search_keyboard_panel"].contains(x, y):
            return True
        button_state.gallery_search_keyboard_open = False
        return True

    if button_state.gallery_viewer_mode in ("image", "video", "grid"):
        if "gallery_delete" in menu_buttons and menu_buttons["gallery_delete"].contains(x, y):
            button_state.gallery_delete_modal_open = True
            return True

    if button_state.gallery_viewer_mode in ("image", "video"):
        if "gallery_prev" in menu_buttons and menu_buttons["gallery_prev"].contains(x, y):
            trigger_viewer_button_feedback("gallery_prev")
            if button_state.gallery_selected_item > 0:
                button_state.gallery_selected_item -= 1
                items = get_displayed_gallery_items(output_dir)

                if button_state.gallery_selected_item < len(items):
                    new_item_type = items[button_state.gallery_selected_item][1]
                    button_state.gallery_viewer_mode = new_item_type

                    if new_item_type == "video":
                        button_state.gallery_video_playing = False
                        button_state.gallery_video_frame_idx = 0
                # Instant transition: no swipe animation
                button_state.gallery_viewer_swipe_offset = 0.0
                button_state.gallery_viewer_swipe_velocity = 0.0
                button_state.gallery_viewer_swipe_inertia_active = False
            return True

        if "gallery_next" in menu_buttons and menu_buttons["gallery_next"].contains(x, y):
            trigger_viewer_button_feedback("gallery_next")
            items = get_displayed_gallery_items(output_dir)
            if button_state.gallery_selected_item < len(items) - 1:
                button_state.gallery_selected_item += 1

                if button_state.gallery_selected_item < len(items):
                    new_item_type = items[button_state.gallery_selected_item][1]
                    button_state.gallery_viewer_mode = new_item_type

                    if new_item_type == "video":
                        button_state.gallery_video_playing = False
                        button_state.gallery_video_frame_idx = 0
                # Instant transition: no swipe animation
                button_state.gallery_viewer_swipe_offset = 0.0
                button_state.gallery_viewer_swipe_velocity = 0.0
                button_state.gallery_viewer_swipe_inertia_active = False
            return True

    if button_state.gallery_viewer_mode == "video":
        if "gallery_play" in menu_buttons and menu_buttons["gallery_play"].contains(x, y):
            trigger_viewer_button_feedback("gallery_play")
            button_state.gallery_video_playing = not button_state.gallery_video_playing
            return True

        if "gallery_progress" in menu_buttons and menu_buttons["gallery_progress"].contains(x, y):
            items = get_displayed_gallery_items(output_dir)
            if button_state.gallery_selected_item is not None and button_state.gallery_selected_item < len(items):
                filepath = items[button_state.gallery_selected_item][0]
                cap = cv2.VideoCapture(str(filepath))
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()

                    progress_btn = menu_buttons["gallery_progress"]
                    click_pos = (x - progress_btn.x) / progress_btn.w
                    click_pos = max(0.0, min(1.0, click_pos))

                    button_state.gallery_video_frame_idx = int(total_frames * click_pos)
                    button_state.gallery_video_playing = False
            return True

    if button_state.gallery_viewer_mode == "grid":

        if "gallery_select_mode" in menu_buttons:
            btn = menu_buttons["gallery_select_mode"]
            if btn.contains(x, y):
                button_state.gallery_select_mode = not button_state.gallery_select_mode
                if not button_state.gallery_select_mode:
                    button_state.gallery_selected_items.clear()

                return True

        if button_state.gallery_select_mode and "gallery_select_all" in menu_buttons:
            if menu_buttons["gallery_select_all"].contains(x, y):
                items = get_displayed_gallery_items(output_dir)
                if items:
                    all_selected = len(button_state.gallery_selected_items) == len(items)
                    if all_selected:
                        button_state.gallery_selected_items.clear()
                    else:
                        button_state.gallery_selected_items = set(range(len(items)))

                return True

        if button_state.gallery_select_mode and "gallery_delete_selected" in menu_buttons:
            if menu_buttons["gallery_delete_selected"].contains(x, y):
                if button_state.gallery_selected_items:
                    button_state.gallery_delete_modal_open = True
                    button_state.gallery_delete_modal_kind = "batch"
                return True

        if hasattr(button_state, 'gallery_thumbnail_rects'):
            for thumb in button_state.gallery_thumbnail_rects:
                if (thumb['x'] <= x <= thumb['x'] + thumb['w'] and
                    thumb['y'] <= y <= thumb['y'] + thumb['h']):
                    idx = thumb['idx']

                    if button_state.gallery_select_mode:
                        if idx in button_state.gallery_selected_items:
                            button_state.gallery_selected_items.remove(idx)
                        else:
                            button_state.gallery_selected_items.add(idx)
                    else:
                        button_state.gallery_selected_item = idx
                        item_type = thumb['type']
                        if item_type == "image":
                            button_state.gallery_viewer_mode = "image"
                        else:
                            button_state.gallery_viewer_mode = "video"
                            button_state.gallery_video_playing = False
                            button_state.gallery_video_frame_idx = 0
                    return True

    return True


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

    if button_state.is_recording and video_recorder is not None:
        rect = get_recording_timestamp_rect()
        if rect is not None:
            rx, ry, rw, rh = rect
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                button_state.is_paused = not button_state.is_paused
                if button_state.is_paused:
                    video_recorder.pause_recording()
                else:
                    video_recorder.resume_recording()
                return video_recorder

    if menu_buttons["menu"].contains(x, y):
        button_state.menu_open = not button_state.menu_open
        return video_recorder

    if not button_state.menu_open:
        return video_recorder

    if "fps30" in menu_buttons and menu_buttons["fps30"].contains(x, y):
        button_state.fps_mode = "30"
        return video_recorder
    if "fps60" in menu_buttons and menu_buttons["fps60"].contains(x, y):
        button_state.fps_mode = "60"
        return video_recorder
    if "fpsmax" in menu_buttons and menu_buttons["fpsmax"].contains(x, y):
        button_state.fps_mode = "MAX"
        return video_recorder

    if "gain" in menu_buttons and menu_buttons["gain"].contains(x, y):
        button_state.gain_mode = "HIGH" if button_state.gain_mode == "LOW" else "LOW"
        menu_buttons["gain"].text = f"GAIN: {button_state.gain_mode}"
        return video_recorder

    if "colormap" in menu_buttons and menu_buttons["colormap"].contains(x, y):
        colormaps = ["MAGMA", "JET", "TURBO", "INFERNO"]
        cur = button_state.colormap_mode
        try:
            i = colormaps.index(cur)
        except ValueError:
            i = 0
        button_state.colormap_mode = colormaps[(i + 1) % len(colormaps)]
        menu_buttons["colormap"].text = f"COLOUR: {button_state.colormap_mode}"
        return video_recorder

    if "cam" in menu_buttons and menu_buttons["cam"].contains(x, y):
        button_state.camera_enabled = not button_state.camera_enabled
        menu_buttons["cam"].text = "CAMERA: ON" if button_state.camera_enabled else "CAMERA: OFF"
        return video_recorder

    if "source" in menu_buttons and menu_buttons["source"].contains(x, y):
        modes = list(SOURCE_MODES)
        cur = button_state.source_mode
        try:
            i = modes.index(cur)
        except ValueError:
            i = modes.index(SOURCE_DEFAULT)
        button_state.source_mode = modes[(i + 1) % len(modes)]
        menu_buttons["source"].text = f"SRC: {button_state.source_mode}"
        return video_recorder

    if "debug" in menu_buttons and menu_buttons["debug"].contains(x, y):
        button_state.debug_enabled = not button_state.debug_enabled
        return video_recorder

    if "shot" in menu_buttons and menu_buttons["shot"].contains(x, y):
        if current_frame is not None and output_dir is not None:
            save_screenshot(current_frame, output_dir)
            button_state.gallery_storage_dirty = True  # so storage bar updates when opening gallery
        return video_recorder

    if "rec" in menu_buttons and menu_buttons["rec"].contains(x, y):
        if video_recorder is None and output_dir is not None:
            video_recorder = VideoRecorder(output_dir, width, height, fps=30)

        if video_recorder is None:
            return None

        button_state.is_recording = not button_state.is_recording
        if button_state.is_recording:
            if video_recorder.start_recording():
                button_state.is_paused = False
                menu_buttons["rec"].text = "STOP"
            else:
                button_state.is_recording = False
        else:
            video_recorder.stop_recording()
            button_state.is_paused = False
            menu_buttons["rec"].text = "REC"
            button_state.gallery_storage_dirty = True  # so storage bar updates when opening gallery
        return video_recorder

    if "gallery" in menu_buttons and menu_buttons["gallery"].contains(x, y):
        button_state.gallery_open = True
        button_state.menu_open = False
        button_state.gallery_storage_dirty = True  # refresh storage bar when opening gallery
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
        modes = list(SOURCE_MODES)
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


def handle_gallery_viewer_mouse(event, x, y, flags, output_dir) -> bool:
    """Handle horizontal swipe/drag with inertia in image or video viewer."""
    if not button_state.gallery_open or button_state.gallery_viewer_mode not in ("image", "video"):
        return False

    now = time.perf_counter()
    items = get_displayed_gallery_items(output_dir)
    n = len(items)
    idx = button_state.gallery_selected_item
    if idx is None or n == 0:
        return False

    # Progress bar scrub (video only): takes priority over swipe on bar
    if event == cv2.EVENT_LBUTTONDOWN and button_state.gallery_viewer_mode == "video":
        if "gallery_progress" in menu_buttons and menu_buttons["gallery_progress"].contains(x, y):
            button_state.gallery_progress_dragging = True
            total_frames = getattr(button_state, "_gallery_video_total_frames", 0)
            if total_frames > 0:
                prog = menu_buttons["gallery_progress"]
                t = (x - prog.x) / prog.w
                t = max(0.0, min(1.0, t))
                button_state.gallery_video_frame_idx = int(t * (total_frames - 1))
            return True

    if event == cv2.EVENT_MOUSEMOVE and button_state.gallery_progress_dragging:
        total_frames = getattr(button_state, "_gallery_video_total_frames", 0)
        if total_frames > 0 and "gallery_progress" in menu_buttons:
            prog = menu_buttons["gallery_progress"]
            t = (x - prog.x) / prog.w
            t = max(0.0, min(1.0, t))
            button_state.gallery_video_frame_idx = int(t * (total_frames - 1))
        return True

    if event == cv2.EVENT_LBUTTONUP and button_state.gallery_progress_dragging:
        button_state.gallery_progress_dragging = False
        return True

    # LBUTTONUP: clear drag state and treat as click if we didn't move (nav/back/delete get priority)
    if event == cv2.EVENT_LBUTTONUP:
        button_state.gallery_viewer_swipe_dragging = False
        if not button_state.gallery_viewer_swipe_drag_moved:
            return handle_gallery_click(x, y, output_dir)
        off = button_state.gallery_viewer_swipe_offset
        vel = button_state.gallery_viewer_swipe_velocity
        th = getattr(button_state, "_gallery_swipe_threshold_px", ui_cache.VIEWER_SWIPE_THRESHOLD_PX)
        vth = ui_cache.VIEWER_SWIPE_VELOCITY_THRESHOLD
        if (off >= th or vel >= vth) and idx < n - 1:
            button_state.gallery_selected_item = idx + 1
            button_state.gallery_viewer_mode = items[idx + 1][1]
            button_state.gallery_viewer_swipe_offset = 0.0
            button_state.gallery_viewer_swipe_velocity = 0.0
            if button_state.gallery_viewer_mode == "video":
                button_state.gallery_video_playing = False
                button_state.gallery_video_frame_idx = 0
            return True
        if (off <= -th or vel <= -vth) and idx > 0:
            button_state.gallery_selected_item = idx - 1
            button_state.gallery_viewer_mode = items[idx - 1][1]
            button_state.gallery_viewer_swipe_offset = 0.0
            button_state.gallery_viewer_swipe_velocity = 0.0
            if button_state.gallery_viewer_mode == "video":
                button_state.gallery_video_playing = False
                button_state.gallery_video_frame_idx = 0
            return True
        if abs(vel) > ui_cache.VIEWER_SWIPE_STOP_VELOCITY:
            button_state.gallery_viewer_swipe_inertia_active = True
            button_state.gallery_viewer_swipe_last_inertia_t = now
        else:
            button_state.gallery_viewer_swipe_offset = 0.0
            button_state.gallery_viewer_swipe_velocity = 0.0
        return True

    # LBUTTONDOWN: don't start swipe if on a button so nav/back/delete take priority
    if event == cv2.EVENT_LBUTTONDOWN:
        on_button = False
        for key in ("gallery_back", "gallery_prev", "gallery_next", "gallery_delete", "gallery_play", "gallery_progress"):
            if key in menu_buttons and menu_buttons[key].contains(x, y):
                on_button = True
                break
        if on_button:
            button_state.gallery_viewer_swipe_drag_moved = False
            return True
        button_state.gallery_viewer_swipe_dragging = True
        button_state.gallery_viewer_swipe_start_x = x
        button_state.gallery_viewer_swipe_offset = 0.0
        button_state.gallery_viewer_swipe_velocity = 0.0
        button_state.gallery_viewer_swipe_last_t = now
        button_state.gallery_viewer_swipe_last_x = x
        button_state.gallery_viewer_swipe_drag_moved = False
        button_state.gallery_viewer_swipe_inertia_active = False
        return True

    if event == cv2.EVENT_MOUSEMOVE and button_state.gallery_viewer_swipe_dragging:
        dx = x - button_state.gallery_viewer_swipe_start_x
        if abs(dx) > ui_cache.DRAG_PX:
            button_state.gallery_viewer_swipe_drag_moved = True
        # Clamp and apply rubber band at first/last item
        max_offset = 120.0
        raw = max(-max_offset, min(max_offset, float(dx)))
        button_state.gallery_viewer_swipe_offset = _viewer_rubber_band_offset(raw, idx, n)
        dt = max(1e-6, now - button_state.gallery_viewer_swipe_last_t)
        inst_v = (x - button_state.gallery_viewer_swipe_last_x) / dt * ui_cache.VIEWER_SWIPE_FLING_GAIN
        button_state.gallery_viewer_swipe_velocity = 0.5 * button_state.gallery_viewer_swipe_velocity + 0.5 * inst_v
        button_state.gallery_viewer_swipe_last_t = now
        button_state.gallery_viewer_swipe_last_x = x
        return True

    return False


def handle_gallery_mouse(event, x, y, flags, output_dir) -> bool:
    if not button_state.gallery_open or button_state.gallery_viewer_mode != "grid":
        return False

    now = time.perf_counter()

    if event == cv2.EVENT_LBUTTONDOWN:
        button_state.gallery_dragging = True
        button_state.gallery_drag_start_y = y
        button_state.gallery_drag_start_x = x
        button_state.gallery_drag_start_scroll = button_state.gallery_scroll_offset
        button_state.gallery_drag_moved = False

        button_state.gallery_scroll_velocity = 0.0
        button_state.gallery_last_drag_t = now
        button_state.gallery_last_drag_y = y
        button_state.gallery_inertia_active = False
        return True

    if event == cv2.EVENT_MOUSEMOVE and button_state.gallery_dragging:
        dy = y - button_state.gallery_drag_start_y
        dx = x - button_state.gallery_drag_start_x

        if (abs(dy) > ui_cache.DRAG_PX) or (abs(dx) > ui_cache.DRAG_PX):
            button_state.gallery_drag_moved = True

        if button_state.gallery_drag_moved:
            new_scroll = button_state.gallery_drag_start_scroll - dy
            max_scroll = int(getattr(button_state, "gallery_max_scroll", 0))
            button_state.gallery_scroll_offset = max(0, min(int(new_scroll), max_scroll))

            dt = max(1e-6, now - button_state.gallery_last_drag_t)
            FLING_GAIN = 2
            EMA_ALPHA = 0.6

            inst_v = (button_state.gallery_last_drag_y - y) / dt
            inst_v *= FLING_GAIN

            button_state.gallery_scroll_velocity = (1.0 - EMA_ALPHA) * button_state.gallery_scroll_velocity + EMA_ALPHA * inst_v
            button_state.gallery_last_drag_t = now
            button_state.gallery_last_drag_y = y

        return True

    if event == cv2.EVENT_LBUTTONUP and button_state.gallery_dragging:
        button_state.gallery_dragging = False

        if not button_state.gallery_drag_moved:
            return handle_gallery_click(x, y, output_dir)

        if abs(button_state.gallery_scroll_velocity) > 50.0:
            button_state.gallery_inertia_active = True
            button_state.gallery_last_inertia_t = now
        else:
            button_state.gallery_inertia_active = False
            button_state.gallery_scroll_velocity = 0.0

        return True

    return False
