"""
Shared icon drawing (WiFi, etc.) used by HUD and menu buttons.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2


def draw_wifi_icon(
    frame: np.ndarray,
    cx: int,
    cy: int,
    *,
    color: Tuple[int, int, int] = (0, 0, 0),
    bg_color: Optional[Tuple[int, int, int]] = (255, 255, 255),
    size: int = 12,
) -> None:
    """
    Draw WiFi icon: arcs (bottom half of circles) + center dot. Used by HUD and menu button.
    HUD: bg_color white, color black (on white pill).
    Menu: bg_color None (transparent), color white (on dark button).
    """
    scale = size / 12.0
    r1, r2, r3 = int(8 * scale), int(5 * scale), max(1, int(2 * scale))
    y1, y2, y3 = int(1 * scale), int(3 * scale), int(5 * scale)
    dot_r = max(1, int(2 * scale))
    if bg_color is not None:
        cv2.circle(frame, (cx, cy), size, bg_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), size, color, 1, cv2.LINE_AA)
    for r, y_off in [(r1, y1), (r2, y2), (r3, y3)]:
        cv2.ellipse(frame, (cx, cy + y_off), (r, r), 0, 180, 360, color, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy + y3), dot_r, color, -1, cv2.LINE_AA)
