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

    # Select-mode tool panels (Tags / Priority / Rename)
    gallery_priority_modal_open: bool = False
    gallery_tags_modal_open: bool = False
    gallery_rename_modal_open: bool = False
    # When > 0, show "Select one or more items first" until time.time() > this (set on Tags/Priority/Rename click with no selection)
    gallery_select_first_hint_until: float = 0.0

    # Modal slide animation: progress 0=hidden, 1=visible; track which modal so we always animate on switch
    gallery_modal_anim_progress: float = 0.0
    gallery_modal_anim_target: float = 0.0
    gallery_modal_anim_start_t: float = 0.0
    gallery_modal_anim_progress_at_flip: float = 0.0
    gallery_modal_anim_active_key: str = ""  # "" when none; forces re-animate when modal changes
    gallery_rename_query: str = ""
    gallery_file_priorities: dict = field(default_factory=dict)  # filename → "high"|"medium"|"low"
    gallery_file_tags: dict = field(default_factory=dict)        # filename → list[str]

    # Tag edit modal (grid select-mode Tags button)
    gallery_tag_modal_open: bool = False
    gallery_tag_active_field: str = ""        # "asset_name" | "asset_type" | "leak_type"
    gallery_tag_keyboard_open: bool = False
    gallery_tag_keyboard_query: str = ""
    gallery_tag_field_values: dict = field(default_factory=dict)  # live values during edit session
    gallery_tag_data: dict = field(default_factory=dict)          # filename → {"asset_type": str, "leak_type": str}
    gallery_tag_info_open: bool = False                           # viewer: read-only info panel

    # Screenshot feedback
    screenshot_flash_time: Optional[float] = None

    # Archive panel (folders for organizing media)
    gallery_archive_folders: list = field(default_factory=list)  # [{"id", "name", "files"}, ...]
    gallery_archive_move_modal_open: bool = False  # Move to folder modal
    gallery_archive_move_hint_until: float = 0.0  # Show "Add folders in Archive panel first"
    gallery_archive_folder_open_id: Optional[str] = None  # folder id when viewing contents popup
    gallery_archive_rename_folder_id: Optional[str] = None  # folder id when renaming
    gallery_archive_rename_query: str = ""  # current rename text for folder

    def __post_init__(self):
        if self.gallery_selected_items is None:
            self.gallery_selected_items = set()
        if self.gallery_file_priorities is None:
            self.gallery_file_priorities = {}
        if self.gallery_file_tags is None:
            self.gallery_file_tags = {}
        if self.gallery_tag_field_values is None:
            self.gallery_tag_field_values = {}
        if self.gallery_tag_data is None:
            self.gallery_tag_data = {}
        if self.gallery_archive_folders is None:
            self.gallery_archive_folders = []


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

# ===============================================================
# UI visibility (top HUD, bottom HUD, menu) – swipe/double-tap to hide/show with animation
# ===============================================================
# Current animated offset (px). Top: 0=visible, negative=hidden up. Bottom: 0=visible, positive=hidden down. Menu: x=right, y=down.
ui_top_hud_offset: float = 0.0
ui_bottom_hud_offset: float = 0.0
ui_menu_offset: float = 0.0
ui_menu_offset_y: float = 0.0
# Target offsets (we lerp toward these each frame)
ui_top_hud_offset_target: float = 0.0
ui_bottom_hud_offset_target: float = 0.0
ui_menu_offset_target: float = 0.0
ui_menu_offset_y_target: float = 0.0
# Gesture state for tap/double-tap and swipe
ui_last_tap_time: float = 0.0
ui_last_tap_x: int = 0
ui_last_tap_y: int = 0
ui_drag_start_x: int = 0
ui_drag_start_y: int = 0
ui_drag_start_time: float = 0.0
ui_drag_handled: bool = False  # True once we've treated this pointer down as a drag
ui_click_was_on_ui: bool = False  # True if LBUTTONDOWN hit menu/HUD/button (gestures ignored on this pointer)