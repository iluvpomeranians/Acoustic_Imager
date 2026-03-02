"""
Button widget, icon drawing, and button/menu registries.
"""

from __future__ import annotations

from typing import Optional, Dict

import cv2
import numpy as np

from . import ui_cache
from ..state import button_state

try:
    from ..config import BUTTON_HITPAD_PX, BUTTON_ALPHA
except Exception:
    BUTTON_HITPAD_PX = ui_cache.BUTTON_HITPAD_PX

buttons: Dict[str, "Button"] = {}
menu_buttons: Dict[str, "Button"] = {}

FPS_MODE_TO_TARGET = {"30": 30, "60": 60}


def _rounded_rect(img, x, y, w, h, r, color, thickness=-1):
    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        color,
        thickness,
        lineType=cv2.LINE_8
    )


def _draw_camera_icon(frame: np.ndarray, cx: int, cy: int, size: int = 12):
    """Draw a camera lens icon."""
    cv2.rectangle(frame, (cx - size, cy - size//2), (cx + size, cy + size//2), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), size//2, (200, 200, 200), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), size//2, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (cx - size, cy - size//2 - 3), (cx - size + 4, cy - size//2), (255, 255, 255), -1, cv2.LINE_AA)


def _draw_rec_icon(frame: np.ndarray, cx: int, cy: int, size: int = 8, is_active: bool = False):
    """Draw a red recording dot icon."""
    color = (0, 0, 255) if is_active else (255, 255, 255)
    cv2.circle(frame, (cx, cy), size, color, -1, cv2.LINE_AA)
    border_color = (255, 255, 255) if is_active else (0, 0, 0)
    cv2.circle(frame, (cx, cy), size, border_color, 1, cv2.LINE_AA)


def _draw_pause_icon(frame: np.ndarray, cx: int, cy: int, size: int = 10):
    """Draw pause bars icon (||)."""
    bar_w = size // 3
    bar_h = size
    gap = size // 3
    cv2.rectangle(frame,
                 (cx - gap - bar_w, cy - bar_h//2),
                 (cx - gap, cy + bar_h//2),
                 (255, 255, 255), -1, cv2.LINE_AA)
    cv2.rectangle(frame,
                 (cx + gap, cy - bar_h//2),
                 (cx + gap + bar_w, cy + bar_h//2),
                 (255, 255, 255), -1, cv2.LINE_AA)


def _draw_back_arrow_icon(frame: np.ndarray, cx: int, cy: int, size: int = 12):
    """Draw a back chevron (<<) so it's clearly not a play triangle."""
    # Left-pointing chevron: two angled segments meeting at the left
    tip_x = cx - size // 2
    top_pt = (cx + size // 2, cy - size)
    bot_pt = (cx + size // 2, cy + size)
    thickness = max(2, size // 5)
    cv2.line(frame, (tip_x, cy), top_pt, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.line(frame, (tip_x, cy), bot_pt, (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_trash_icon(frame: np.ndarray, cx: int, cy: int, size: int = 12):
    """Draw a trash can icon: body with trapezoid shape, lid, and arched handle."""
    body_w = int(size * 1.0)
    body_h = int(size * 1.2)
    top_y = cy - body_h // 2
    bottom_y = cy + body_h // 2
    # Body: trapezoid (wider at bottom) for depth
    body_top_w = body_w
    body_bot_w = int(body_w * 1.15)
    pts = np.array([
        [cx - body_top_w // 2, top_y + 4],
        [cx + body_top_w // 2, top_y + 4],
        [cx + body_bot_w // 2, bottom_y - 2],
        [cx - body_bot_w // 2, bottom_y - 2],
    ], np.int32)
    cv2.fillPoly(frame, [pts], (255, 255, 255), cv2.LINE_AA)
    cv2.polylines(frame, [pts], True, (180, 180, 180), 1, cv2.LINE_AA)
    # Lid (rounded top)
    lid_h = 3
    cv2.rectangle(frame,
                  (cx - body_top_w // 2, top_y),
                  (cx + body_top_w // 2, top_y + lid_h),
                  (255, 255, 255), -1, cv2.LINE_AA)
    # Handle: small arch above lid
    handle_w = body_w // 2
    handle_top = top_y - 2
    cv2.ellipse(frame, (cx, handle_top + 2), (handle_w // 2, 3), 0, 180, 360, (255, 255, 255), 2, cv2.LINE_AA)


class Button:
    def __init__(self, x, y, w, h, text):
        self.x, self.y, self.w, self.h = int(x), int(y), int(w), int(h)
        self.text = text
        self.is_hovered = False
        self.is_active = False

    def contains(self, mx, my) -> bool:
        pad = BUTTON_HITPAD_PX
        return (
            (self.x - pad) <= mx <= (self.x + self.w + pad) and
            (self.y - pad) <= my <= (self.y + self.h + pad)
        )

    def draw(self, frame: np.ndarray, transparent: bool = False, active_color: Optional[tuple] = None, active_border_color: Optional[tuple] = None, icon_type: Optional[str] = None) -> None:
        base = (60, 60, 60)
        hover = (85, 85, 85)
        active = active_color if active_color is not None else (40, 200, 60)

        color = active if self.is_active else (hover if self.is_hovered else base)

        x, y, w, h = self.x, self.y, self.w, self.h

        if transparent:
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(frame.shape[1], x + w)
            y1 = min(frame.shape[0], y + h)

            if x1 > x0 and y1 > y0:
                roi = frame[y0:y1, x0:x1]
                overlay = np.empty_like(roi)
                overlay[:] = color
                alpha = BUTTON_ALPHA
                cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)
        else:
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(frame.shape[1], x + w)
            y1 = min(frame.shape[0], y + h)
            if x1 > x0 and y1 > y0:
                roi = frame[y0:y1, x0:x1]
                roi[:] = ui_cache.get_grad(roi.shape[1], roi.shape[0], color)

        if self.is_active:
            if active_border_color is not None:
                border_color = active_border_color
            elif active_color is not None:
                border_color = tuple(min(255, int(c * 1.3)) for c in active_color)
            else:
                border_color = (80, 255, 100)
        else:
            border_color = (255, 255, 255)
        _rounded_rect(frame, x, y, w, h, r=10, color=border_color, thickness=2)

        if icon_type:
            cx = x + w // 2
            cy = y + h // 2

            if icon_type == "camera":
                _draw_camera_icon(frame, cx, cy, size=10)
            elif icon_type == "rec":
                _draw_rec_icon(frame, cx, cy, size=7, is_active=self.is_active)
            elif icon_type == "pause":
                _draw_pause_icon(frame, cx, cy, size=9)
            elif icon_type == "back":
                _draw_back_arrow_icon(frame, cx, cy, size=12)
            elif icon_type == "trash":
                _draw_trash_icon(frame, cx, cy, size=14)
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.42 if "LOOP" in self.text else 0.52
            thick = 1
            tw, th = cv2.getTextSize(self.text, font, scale, thick)[0]
            tx = x + (w - tw) // 2
            ty = y + (h + th) // 2

            # cv2.putText(frame, self.text, (tx, ty),
            #             font, scale, (0, 0, 0),
            #             thick + 1, cv2.LINE_AA)

            cv2.putText(frame, self.text, (tx, ty),
                        font, scale, (255, 255, 255),
                        thick, cv2.LINE_AA)


def init_buttons(left_width: int, camera_available: bool) -> None:
    buttons.clear()


def init_menu_buttons(left_width: int, frame_height: int = None) -> None:
    menu_buttons.clear()

    actual_height = frame_height if frame_height is not None else ui_cache.HEIGHT

    menu_w = 180
    menu_h = 50
    menu_margin_x = 15
    menu_margin_bottom = 5
    menu_x = left_width - menu_w - menu_margin_x
    menu_y = actual_height - menu_h - menu_margin_bottom

    menu_buttons["menu"] = Button(menu_x, menu_y, menu_w, menu_h, "MENU")

    item_h = 40
    gap = 8

    total_items = 7
    dropdown_h = total_items * (item_h + gap) + gap

    dropdown_y = menu_y - dropdown_h - gap
    y0 = dropdown_y

    seg_gap = 6
    seg_w = (menu_w - 2 * seg_gap) // 3
    menu_buttons["fps30"]  = Button(menu_x + 0 * (seg_w + seg_gap), y0, seg_w, item_h, "30FPS")
    menu_buttons["fps60"]  = Button(menu_x + 1 * (seg_w + seg_gap), y0, seg_w, item_h, "60FPS")
    menu_buttons["fpsmax"] = Button(menu_x + 2 * (seg_w + seg_gap), y0, seg_w, item_h, "MAX")

    gain_y = y0 + (item_h + gap)
    menu_buttons["gain"] = Button(menu_x, gain_y, menu_w, item_h, f"GAIN: {button_state.gain_mode}")

    colormap_y = gain_y + (item_h + gap)
    menu_buttons["colormap"] = Button(menu_x, colormap_y, menu_w, item_h, f"COLOUR: {button_state.colormap_mode}")

    cam_y = colormap_y + (item_h + gap)
    menu_buttons["cam"] = Button(
        menu_x, cam_y, menu_w, item_h,
        "CAMERA: ON" if button_state.camera_enabled else "CAMERA: OFF"
    )

    src_y = cam_y + (item_h + gap)
    tools_y = src_y + (item_h + gap)
    tool_gap = 6
    tool_w = (menu_w - tool_gap) // 2
    menu_buttons["source"] = Button(menu_x + 0 * (tool_w + tool_gap), src_y, tool_w, item_h, f"SRC: {button_state.source_mode}")
    menu_buttons["debug"] = Button(menu_x + 1 * (tool_w + tool_gap), src_y, tool_w, item_h, "DEBUG")
    menu_buttons["shot"]  = Button(menu_x + 0 * (tool_w + tool_gap), tools_y, tool_w, item_h, "SHOT")
    menu_buttons["rec"]   = Button(menu_x + 1 * (tool_w + tool_gap), tools_y, tool_w, item_h, "REC")

    gallery_y = tools_y + (item_h + gap)
    menu_buttons["gallery"] = Button(menu_x, gallery_y, menu_w, item_h, "GALLERY")


def update_button_states(mx: int, my: int) -> None:
    for b in buttons.values():
        b.is_hovered = b.contains(mx, my)

    if "menu" in menu_buttons:
        menu_buttons["menu"].is_hovered = menu_buttons["menu"].contains(mx, my)

    keys = ("fps30", "fps60", "fpsmax", "gain", "colormap", "cam", "source", "debug", "shot", "rec", "gallery")

    if button_state.menu_open:
        for k in keys:
            if k in menu_buttons:
                menu_buttons[k].is_hovered = menu_buttons[k].contains(mx, my)
    else:
        for k in keys:
            if k in menu_buttons:
                menu_buttons[k].is_hovered = False

    if button_state.gallery_open:
        gallery_keys = ("gallery_back", "gallery_select_mode", "gallery_select_all", "gallery_delete_selected",
                       "gallery_delete", "gallery_prev", "gallery_next", "gallery_play", "gallery_progress")
        for k in gallery_keys:
            if k in menu_buttons:
                menu_buttons[k].is_hovered = menu_buttons[k].contains(mx, my)

    if button_state.gallery_delete_modal_open:
        for k in ("modal_yes", "modal_no"):
            if k in menu_buttons:
                menu_buttons[k].is_hovered = menu_buttons[k].contains(mx, my)


def draw_buttons(frame: np.ndarray) -> None:
    for b in buttons.values():
        b.draw(frame)
