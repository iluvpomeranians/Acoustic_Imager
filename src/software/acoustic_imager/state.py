# state.py
"""
Runtime state holders for the Acoustic Imager software.

This keeps "mutable globals" out of config.py and mirrors the defaults
from the original monolithic script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Any

from .config import USE_CAMERA, F_MIN_HZ_DEFAULT, F_MAX_HZ_DEFAULT, SOURCE_DEFAULT

# ===============================================================
# HUD state
# ===============================================================
@dataclass
class HudState:
    details_level: str = "MIN"   # "OFF" | "MIN" | "MAX"
    open_panel: str = ""         # "" | "time" | "fps" | "net"

HUD = HudState()


# ===============================================================
# UI / App state
# ===============================================================
@dataclass
class ButtonState:
    is_recording: bool = False
    is_paused: bool = False
    camera_enabled: bool = USE_CAMERA
    source_mode: str = SOURCE_DEFAULT  # "SIM" | "SPI_LOOPBACK" | "SPI_HW"

    # MENU states
    menu_open: bool = False
    fps_mode: str = "MAX"       # "30" | "60" | "MAX"
    gain_mode: str = "LOW"     # placeholder toggle
    debug_enabled: bool = True
    colormap_mode: str = "MAGMA"  # "MAGMA" | "JET" | "TURBO" | "INFERNO"

    # Gallery state
    gallery_open: bool = False
    gallery_scroll_offset: int = 0
    gallery_selected_item: Optional[int] = None
    gallery_viewer_mode: str = "grid"  # "grid" | "image" | "video"
    gallery_video_playing: bool = False
    gallery_video_frame_idx: int = 0
    gallery_dragging = False
    gallery_drag_start_y: int = 0
    gallery_drag_start_scroll: int = 0
    gallery_drag_moved = False
    gallery_drag_start_x = 0
    gallery_select_mode: bool = False  # Whether we're in select mode for multi-select
    gallery_selected_items: set[int] = field(default_factory=set)  # Multi-select for deletion
    gallery_delete_confirm: bool = False  # Confirmation state for delete
    gallery_delete_modal_open: bool = False  # Whether delete confirmation modal is open
    gallery_scroll_velocity: float = 0.0   # px/s
    gallery_last_drag_t: float = 0.0
    gallery_last_drag_y: int = 0
    gallery_inertia_active: bool = False
    gallery_last_inertia_t: float = 0.0

    # Viewer (image/video) horizontal swipe with inertia
    gallery_viewer_swipe_dragging: bool = False
    gallery_viewer_swipe_start_x: int = 0
    gallery_viewer_swipe_offset: float = 0.0   # px, positive = dragged left (toward next)
    gallery_viewer_swipe_velocity: float = 0.0  # px/s
    gallery_viewer_swipe_last_t: float = 0.0
    gallery_viewer_swipe_last_x: int = 0
    gallery_viewer_swipe_drag_moved: bool = False
    gallery_viewer_swipe_inertia_active: bool = False
    gallery_viewer_swipe_last_inertia_t: float = 0.0

    # Video progress bar scrub (set by draw, read by handler)
    gallery_progress_dragging: bool = False

    # Viewer dock: green click feedback (button key + time)
    viewer_dock_feedback_button: str = ""
    viewer_dock_feedback_time: float = 0.0

    gallery_delete_modal_open: bool = False
    gallery_delete_modal_kind: str = "single"
    gallery_delete_modal_title: str = "Delete this item?"
    gallery_delete_modal_subtitle: str = "This action cannot be undone."
    gallery_storage_dirty: bool = False  # set when file(s) deleted so storage bar syncs and updates

    # Gallery dock: filter (type), sort, search
    gallery_filter_type: str = "all"  # "all" | "image" | "video"
    gallery_sort_by: str = "date"  # "date" | "name" | "size"
    gallery_search_query: str = ""
    gallery_filter_modal_open: bool = False
    gallery_sort_modal_open: bool = False
    gallery_search_keyboard_open: bool = False

    # Screenshot feedback
    screenshot_flash_time: Optional[float] = None

    def __post_init__(self):
        if self.gallery_selected_items is None:
            self.gallery_selected_items = set()


# A single shared instance (matches original `button_state = ButtonState()`)
button_state = ButtonState()


# ===============================================================
# Bandpass drag state
# ===============================================================
DRAG_ACTIVE: bool = False
DRAG_TARGET: Optional[str] = None  # "min", "max", or "box"
DRAG_START_Y: int = 0  # Initial y position when box drag starts
DRAG_START_F_MIN: float = 0.0  # Initial F_MIN_HZ when box drag starts
DRAG_START_F_MAX: float = 0.0  # Initial F_MAX_HZ when box drag starts

F_MIN_HZ: float = float(F_MIN_HZ_DEFAULT)
F_MAX_HZ: float = float(F_MAX_HZ_DEFAULT)


# ===============================================================
# Mouse + shared frame/output handles
# ===============================================================
CURSOR_POS: Tuple[int, int] = (0, 0)

CURRENT_FRAME: Optional[Any] = None  # typically a numpy ndarray (H,W,3) uint8
OUTPUT_DIR: Optional[Path] = None

CAMERA_AVAILABLE: bool = False

# ===============================================================
# Recording timestamp tracking
# ===============================================================
RECORDING_START_TIME: Optional[float] = None  # time.time() when recording started
RECORDING_PAUSED_TIME: Optional[float] = None  # time.time() when paused
RECORDING_TOTAL_PAUSED: float = 0.0  # total time spent paused