# state.py
"""
Runtime state holders for the Acoustic Imager software.

This keeps "mutable globals" out of config.py and mirrors the defaults
from the original monolithic script.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    gallery_drag_active: bool = False
    gallery_drag_start_y: int = 0
    gallery_drag_start_offset: int = 0
    gallery_select_mode: bool = False  # Whether we're in select mode for multi-select
    gallery_selected_items: set = None  # Multi-select for deletion
    gallery_delete_confirm: bool = False  # Confirmation state for delete
    gallery_delete_modal_open: bool = False  # Whether delete confirmation modal is open
    
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
DRAG_TARGET: Optional[str] = None  # "min" or "max"

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