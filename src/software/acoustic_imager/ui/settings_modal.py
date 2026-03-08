"""
Settings modal: blank centered rectangular popup (placeholder for future settings).
Matches size of WiFi modal.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .button import menu_buttons, Button
from ..state import HUD

# Match WiFi modal dimensions (password screen size)
MODAL_W = 520
MODAL_H = 420


def draw_settings_modal(frame: np.ndarray) -> None:
    """Draw Settings modal: dark overlay + centered blank rectangle."""
    if not HUD.settings_modal_open:
        return

    fh, fw = frame.shape[:2]
    border_color = (100, 100, 100)

    # Dark overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Centered rectangular popup (same size as WiFi modal)
    modal_x = (fw - MODAL_W) // 2
    modal_y = (fh - MODAL_H) // 2

    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + MODAL_W, modal_y + MODAL_H), (40, 40, 40), -1)
    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + MODAL_W, modal_y + MODAL_H), border_color, 3, cv2.LINE_AA)

    # Register panel for click-outside-to-close
    if "settings_modal_panel" not in menu_buttons:
        menu_buttons["settings_modal_panel"] = Button(modal_x, modal_y, MODAL_W, MODAL_H, "")
    else:
        b = menu_buttons["settings_modal_panel"]
        b.x, b.y, b.w, b.h = modal_x, modal_y, MODAL_W, MODAL_H


def handle_settings_modal_click(x: int, y: int) -> bool:
    """Handle click: close if clicking outside the modal panel. Returns True if handled."""
    if not HUD.settings_modal_open:
        return False

    if "settings_modal_panel" in menu_buttons:
        panel = menu_buttons["settings_modal_panel"]
        if panel.contains(x, y):
            return True
    HUD.settings_modal_open = False
    return True
