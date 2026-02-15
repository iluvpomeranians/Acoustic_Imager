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

from .config import USE_CAMERA, F_MIN_HZ_DEFAULT, F_MAX_HZ_DEFAULT


# ===============================================================
# UI / App state
# ===============================================================
@dataclass
class ButtonState:
    is_recording: bool = False
    is_paused: bool = False
    camera_enabled: bool = USE_CAMERA
    source_mode: str = "SIM"   # "SIM" or "SPI"

    # MENU states
    menu_open: bool = False
    fps_mode: str = "MAX"       # "30" | "60" | "MAX"
    gain_mode: str = "LOW"     # placeholder toggle
    debug_enabled: bool = True


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