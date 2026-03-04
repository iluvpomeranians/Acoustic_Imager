"""
Grid side dock: vertical strip on the right in gallery grid view.

Dock is delimited from the grid; the storage bar floats with the viewport
(stays visible) as you scroll up/down.
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple
import shutil

import cv2
import numpy as np

from ..config import MENU_ACTIVE_BLUE, MENU_ACTIVE_BLUE_LIGHT, MODAL_ACTIVE_GOLD, DOCK_GRADIENT_TOP, DOCK_GRADIENT_BOT
from ..state import button_state
from .button import Button, menu_buttons
from .keyboard import (
    ROWS_QWERTY as KEYBOARD_ROWS_QWERTY,
    ROW_NUMBERS as KEYBOARD_ROW_NUMBERS,
    SPECIAL_KEYS_FULL as KEYBOARD_SPECIAL,
    FULL_KEY_SCALE as KEY_SCALE,
    draw_key_bg_clipped,
    dimensions_for_scale,
    KEY_BORDER_BGR as KEYBOARD_KEY_BORDER,
    KEY_TEXT_BGR as KEYBOARD_KEY_TEXT,
)
from .priority_circle import draw_priority_circle_neon

GRID_SIDE_DOCK_WIDTH = 113

DOCK_BG = (22, 22, 26)
DOCK_EDGE = (55, 55, 60)
DOCK_EDGE_ACCENT = (75, 75, 82)
DOCK_INNER_PAD = 12

# Top dock rows: Search, Filter, and third — flush with inset, dividers between
DOCK_TOP_INSET_X = 2
DOCK_TOP_INSET_Y = 0
DOCK_ROW_HEIGHT = 75  # 50% larger than 50
DOCK_DIVIDER_COLOR = (95, 95, 105)
DOCK_DIVIDER_THICKNESS = 2
# Row labels (third is generic; change as needed)
SEARCH_BAR_PLACEHOLDER = "Search"
DOCK_ROW_FILTER_LABEL = "Filter"
DOCK_ROW_THIRD_LABEL = "Sort"
SEARCH_BAR_TEXT_COLOR = (255, 255, 255)

BAR_BOTTOM_MARGIN_PX = 2
BAR_HEIGHT = 200
BAR_INSET = 2
# Bar: width and margins (centered between text and dock right edge)
BAR_WIDTH_PX = 22
BAR_RIGHT_MARGIN_PX = 4
BAR_TEXT_GAP_PX = 8  # gap between end of text and bar

# Priority colours (BGR)
PRIORITY_COLORS = {
    "high":   (30,  30, 220),   # red
    "medium": (0,  140, 255),   # orange
    "low":    (50, 185,  50),   # green
}
PRIORITY_OPTIONS = [("High", "high"), ("Medium", "medium"), ("Low", "low")]

# Preset tags shown in the tags panel
PRESET_TAGS = ["Important", "Reviewed", "Processed", "Follow-up", "Archive"]


def _vertical_gradient(h: int, w: int, top_bgr: Tuple[int, int, int], bot_bgr: Tuple[int, int, int]) -> np.ndarray:
    """Vertical gradient (top -> bottom) as (h, w, 3) BGR uint8."""
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        out[:, :, c] = np.linspace(top_bgr[c], bot_bgr[c], h, dtype=np.uint8).reshape(-1, 1)
    return out


def _draw_icon_search(frame: np.ndarray, cx: int, cy: int, color: Tuple[int, int, int]) -> None:
    """Draw magnifying glass icon centered at (cx, cy) — 50% larger."""
    icon_r = 9
    cv2.circle(frame, (cx, cy), icon_r, color, 1, cv2.LINE_AA)
    cv2.line(
        frame,
        (cx + 6, cy + 3),
        (cx + 16, cy + 13),
        color, 1, cv2.LINE_AA
    )


def _draw_icon_filter(frame: np.ndarray, cx: int, cy: int, color: Tuple[int, int, int]) -> None:
    """Draw funnel/filter icon (trapezoid) centered at (cx, cy) — 50% larger."""
    pts = np.array([
        [cx - 12, cy - 7],
        [cx + 12, cy - 7],
        [cx + 6, cy + 7],
        [cx - 6, cy + 7],
    ], dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)


def _draw_icon_sort(frame: np.ndarray, cx: int, cy: int, color: Tuple[int, int, int]) -> None:
    """Draw stacked lines (ascending) icon for sort — 50% larger."""
    cv2.line(frame, (cx - 7, cy - 7), (cx + 5, cy - 7), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx - 7, cy), (cx + 7, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx - 7, cy + 7), (cx + 12, cy + 7), color, 1, cv2.LINE_AA)


def _draw_icon_tag(frame: np.ndarray, cx: int, cy: int, color: Tuple[int, int, int]) -> None:
    """Draw a price-tag / label icon centered at (cx, cy)."""
    pts = np.array([
        [cx - 11, cy - 7],
        [cx + 3,  cy - 7],
        [cx + 11, cy],
        [cx + 3,  cy + 7],
        [cx - 11, cy + 7],
    ], dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (cx - 6, cy), 2, color, -1, cv2.LINE_AA)


def _draw_icon_priority(frame: np.ndarray, cx: int, cy: int, color: Tuple[int, int, int]) -> None:
    """Draw a flag icon for priority centered at (cx, cy)."""
    cv2.line(frame, (cx - 6, cy - 10), (cx - 6, cy + 10), color, 2, cv2.LINE_AA)
    flag_pts = np.array([
        [cx - 6, cy - 10],
        [cx + 9, cy - 4],
        [cx - 6, cy + 2],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [flag_pts], color)


def _draw_icon_pen(frame: np.ndarray, cx: int, cy: int, color: Tuple[int, int, int]) -> None:
    """Draw a simple pen icon: vertical body + pointed nib at bottom."""
    # Body (vertical rectangle, cap at top)
    body_top = cy - 10
    body_bot = cy + 5
    body_left = cx - 3
    body_right = cx + 3
    cv2.rectangle(frame, (body_left, body_top), (body_right, body_bot), color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (body_left, body_top), (body_right, body_bot), color, 1, cv2.LINE_AA)
    # Nib (small triangle pointing down at writing end)
    tip_pts = np.array([
        [body_left, body_bot],
        [body_right, body_bot],
        [cx, cy + 10],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [tip_pts], color)
    cv2.polylines(frame, [tip_pts], True, color, 1, cv2.LINE_AA)


# Subtle border for dock buttons so they read as controls, not flat panels
DOCK_ROW_BORDER = (100, 100, 110)
DOCK_ROW_TOP_HIGHLIGHT = (140, 120, 90)
# White outline for button+modal when modal is open (continuous extension)
DOCK_ROW_WHITE_BORDER = (255, 255, 255)
MODAL_ANIM_DURATION_S = 0.28


def _draw_dock_row(
    frame: np.ndarray,
    x0: int,
    y0: int,
    w: int,
    h: int,
    label: str,
    icon_right: Optional[str] = None,
    skip_bg: bool = False,
) -> None:
    """Draw one top-dock row: gradient fill (unless skip_bg), subtle border, icon above text, centered."""
    if w < 4 or h < 8:
        return
    if not skip_bg:
        roi = frame[y0:y0 + h, x0:x0 + w]
        gradient = _vertical_gradient(h, w, MENU_ACTIVE_BLUE, MENU_ACTIVE_BLUE_LIGHT)
        roi[:] = gradient
        cv2.rectangle(frame, (x0, y0), (x0 + w - 1, y0 + h - 1), DOCK_ROW_WHITE_BORDER, 1, cv2.LINE_AA)
        cv2.line(frame, (x0, y0), (x0 + w, y0), DOCK_ROW_TOP_HIGHLIGHT, 1, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.71
    (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
    gap = 6
    icon_half_h = 10
    block_h = 2 * icon_half_h + gap + th
    block_top = y0 + (h - block_h) // 2
    block_cx = x0 + w // 2
    icon_cx = block_cx
    icon_cy = block_top + icon_half_h
    text_x = block_cx - tw // 2
    text_y = block_top + 2 * icon_half_h + gap + th
    if icon_right == "search":
        _draw_icon_search(frame, icon_cx, icon_cy, SEARCH_BAR_TEXT_COLOR)
    elif icon_right == "filter":
        _draw_icon_filter(frame, icon_cx, icon_cy, SEARCH_BAR_TEXT_COLOR)
    elif icon_right == "sort":
        _draw_icon_sort(frame, icon_cx, icon_cy, SEARCH_BAR_TEXT_COLOR)
    elif icon_right == "tag":
        _draw_icon_tag(frame, icon_cx, icon_cy, SEARCH_BAR_TEXT_COLOR)
    elif icon_right == "priority":
        _draw_icon_priority(frame, icon_cx, icon_cy, SEARCH_BAR_TEXT_COLOR)
    elif icon_right == "pen":
        _draw_icon_pen(frame, icon_cx, icon_cy, SEARCH_BAR_TEXT_COLOR)
    cv2.putText(
        frame, label, (text_x, text_y),
        font, scale, SEARCH_BAR_TEXT_COLOR, 1, cv2.LINE_AA
    )


def _draw_dock_top_rows(
    frame: np.ndarray,
    dock_x: int,
    dock_y: int,
    dock_w: int,
    active_modal_key: Optional[str] = None,
    progress: float = 0.0,
) -> None:
    """Draw top dock rows.

    Normal mode  → Search / Filter / Sort
    Select mode  → Tags / Priority / Rename
    When active_modal_key is set and progress > 0, that row skips its own bg (combined strip drawn separately).
    """
    if button_state.gallery_select_mode:
        _draw_select_mode_rows(frame, dock_x, dock_y, dock_w, active_modal_key, progress)
        return

    inset_x = DOCK_TOP_INSET_X
    inset_y = DOCK_TOP_INSET_Y
    row_h = DOCK_ROW_HEIGHT
    x0 = dock_x + inset_x
    w = dock_w - 2 * inset_x
    if w < 10:
        return
    y = dock_y + inset_y
    skip = active_modal_key == "search" and progress > 0
    # Row 1: Search
    _draw_dock_row(frame, x0, y, w, row_h, SEARCH_BAR_PLACEHOLDER, icon_right="search", skip_bg=skip)
    if "gallery_dock_search" not in menu_buttons:
        menu_buttons["gallery_dock_search"] = Button(x0, y, w, row_h, "Search")
    else:
        b = menu_buttons["gallery_dock_search"]
        b.x, b.y, b.w, b.h = x0, y, w, row_h
    y += row_h
    for _ in range(DOCK_DIVIDER_THICKNESS):
        cv2.line(frame, (dock_x, y), (dock_x + dock_w, y), DOCK_DIVIDER_COLOR, 1, cv2.LINE_AA)
        y += 1
    skip = active_modal_key == "filter" and progress > 0
    # Row 2: Filter
    _draw_dock_row(frame, x0, y, w, row_h, DOCK_ROW_FILTER_LABEL, icon_right="filter", skip_bg=skip)
    if "gallery_dock_filter" not in menu_buttons:
        menu_buttons["gallery_dock_filter"] = Button(x0, y, w, row_h, "Filter")
    else:
        b = menu_buttons["gallery_dock_filter"]
        b.x, b.y, b.w, b.h = x0, y, w, row_h
    y += row_h
    for _ in range(DOCK_DIVIDER_THICKNESS):
        cv2.line(frame, (dock_x, y), (dock_x + dock_w, y), DOCK_DIVIDER_COLOR, 1, cv2.LINE_AA)
        y += 1
    skip = active_modal_key == "sort" and progress > 0
    # Row 3: Sort
    _draw_dock_row(frame, x0, y, w, row_h, DOCK_ROW_THIRD_LABEL, icon_right="sort", skip_bg=skip)
    if "gallery_dock_sort" not in menu_buttons:
        menu_buttons["gallery_dock_sort"] = Button(x0, y, w, row_h, "Sort")
    else:
        b = menu_buttons["gallery_dock_sort"]
        b.x, b.y, b.w, b.h = x0, y, w, row_h


def _draw_select_mode_rows(
    frame: np.ndarray,
    dock_x: int,
    dock_y: int,
    dock_w: int,
    active_modal_key: Optional[str] = None,
    progress: float = 0.0,
) -> None:
    """Draw Tags / Priority / Rename rows shown when gallery_select_mode is active."""
    inset_x = DOCK_TOP_INSET_X
    inset_y = DOCK_TOP_INSET_Y
    row_h = DOCK_ROW_HEIGHT
    x0 = dock_x + inset_x
    w = dock_w - 2 * inset_x
    if w < 10:
        return
    y = dock_y + inset_y

    skip = active_modal_key == "tags" and progress > 0
    # Row 1: Tags
    _draw_dock_row(frame, x0, y, w, row_h, "Tags", icon_right="tag", skip_bg=skip)
    if "gallery_dock_tags" not in menu_buttons:
        menu_buttons["gallery_dock_tags"] = Button(x0, y, w, row_h, "Tags")
    else:
        b = menu_buttons["gallery_dock_tags"]
        b.x, b.y, b.w, b.h = x0, y, w, row_h
    y += row_h
    for _ in range(DOCK_DIVIDER_THICKNESS):
        cv2.line(frame, (dock_x, y), (dock_x + dock_w, y), DOCK_DIVIDER_COLOR, 1, cv2.LINE_AA)
        y += 1

    skip = active_modal_key == "priority" and progress > 0
    # Row 2: Priority
    _draw_dock_row(frame, x0, y, w, row_h, "Priority", icon_right="priority", skip_bg=skip)
    if "gallery_dock_priority" not in menu_buttons:
        menu_buttons["gallery_dock_priority"] = Button(x0, y, w, row_h, "Priority")
    else:
        b = menu_buttons["gallery_dock_priority"]
        b.x, b.y, b.w, b.h = x0, y, w, row_h
    y += row_h
    for _ in range(DOCK_DIVIDER_THICKNESS):
        cv2.line(frame, (dock_x, y), (dock_x + dock_w, y), DOCK_DIVIDER_COLOR, 1, cv2.LINE_AA)
        y += 1

    skip = active_modal_key == "rename" and progress > 0
    # Row 3: Rename
    _draw_dock_row(frame, x0, y, w, row_h, "Rename", icon_right="pen", skip_bg=skip)
    if "gallery_dock_rename" not in menu_buttons:
        menu_buttons["gallery_dock_rename"] = Button(x0, y, w, row_h, "Rename")
    else:
        b = menu_buttons["gallery_dock_rename"]
        b.x, b.y, b.w, b.h = x0, y, w, row_h


# Filter modal: option labels and state values
FILTER_OPTIONS = [("All", "all"), ("Images", "image"), ("Videos", "video")]
# Sort modal: option labels and state values
SORT_OPTIONS = [
    ("Date (newest)", "date"),
    ("Name (A to Z)", "name"),
    ("Size (largest)", "size"),
    ("Priority (Highest)", "priority"),
]

MODAL_PANEL_W = 320
MODAL_PANEL_H = 200  # used for filter modal (3 options)
# Edit Tags form in gallery.py uses this width
TAGS_FORM_W = 590
MODAL_OPTION_H = 44
MODAL_TITLE_H = 50

# Search & Rename keyboards: dimensions from keyboard module (identical size, gray keys)
_KEYBOARD_DIMS = dimensions_for_scale(KEY_SCALE)
KEY_W = _KEYBOARD_DIMS["key_w"]
KEY_H = _KEYBOARD_DIMS["key_h"]
KEY_GAP = _KEYBOARD_DIMS["key_gap"]
KEYBOARD_BAR_H = _KEYBOARD_DIMS["bar_h"]
KEYBOARD_FOOTER_GAP = _KEYBOARD_DIMS["footer_gap"]
KEYBOARD_FONT_BAR = _KEYBOARD_DIMS["font_bar"]
KEYBOARD_FONT_KEY = _KEYBOARD_DIMS["font_key"]
KEYBOARD_FONT_SPECIAL = _KEYBOARD_DIMS["font_special"]
KEYBOARD_MARGIN_BOTTOM = 24


# Modal as extension of dock: gap west of dock, no overlay.
# Connector width so modal blue meets the row (continuous from button to modal).
MODAL_GAP_WEST = 8
MODAL_CONNECTOR_WIDTH = MODAL_GAP_WEST + DOCK_TOP_INSET_X  # 10px: modal extends to row start


def _get_active_modal_info(
    frame: np.ndarray, dock_x: int, dock_y: int, dock_w: int
) -> Optional[Tuple[str, int, int, int, int, int]]:
    """Return (active_key, row_y, row_h, modal_px, modal_total_w, modal_h) for the open modal, or None."""
    fh = frame.shape[0]
    row_h = DOCK_ROW_HEIGHT
    inset_y = DOCK_TOP_INSET_Y
    div = DOCK_DIVIDER_THICKNESS

    if button_state.gallery_select_mode:
        if button_state.gallery_tag_modal_open:
            row_y = dock_y + inset_y
            px = dock_x - MODAL_GAP_WEST - TAGS_FORM_W
            total_w = TAGS_FORM_W + MODAL_CONNECTOR_WIDTH
            return ("tags", row_y, row_h, px, total_w, 280)
        if button_state.gallery_priority_modal_open:
            panel_h = MODAL_TITLE_H + len(PRIORITY_OPTIONS) * MODAL_OPTION_H + 10
            row_y = dock_y + inset_y + row_h + div
            px = dock_x - MODAL_GAP_WEST - MODAL_PANEL_W
            total_w = MODAL_PANEL_W + MODAL_CONNECTOR_WIDTH
            return ("priority", row_y, row_h, px, total_w, panel_h)
        if button_state.gallery_rename_modal_open:
            num_letter_rows = len(KEYBOARD_ROWS_QWERTY)
            num_rows = num_letter_rows + 1 + 1
            total_key_h = num_rows * (KEY_H + KEY_GAP) + KEY_GAP
            panel_h = KEYBOARD_BAR_H + total_key_h + KEYBOARD_FOOTER_GAP
            row_y = dock_y + inset_y + 2 * (row_h + div)
            max_letters = max(len(r) for r in KEYBOARD_ROWS_QWERTY)
            max_row_w = max(max_letters, len(KEYBOARD_ROW_NUMBERS)) * (KEY_W + KEY_GAP) + KEY_GAP
            special_w = KEY_W * 2
            special_row_w = len(KEYBOARD_SPECIAL) * (special_w + KEY_GAP) + KEY_GAP
            panel_w = max(max_row_w, special_row_w, int(220 * KEY_SCALE))
            px = dock_x - MODAL_GAP_WEST - panel_w
            total_w = panel_w + MODAL_CONNECTOR_WIDTH
            return ("rename", row_y, row_h, px, total_w, panel_h)
    else:
        if button_state.gallery_search_keyboard_open:
            num_letter_rows = len(KEYBOARD_ROWS_QWERTY)
            num_rows = num_letter_rows + 1 + 1
            total_key_h = num_rows * (KEY_H + KEY_GAP) + KEY_GAP
            panel_h = KEYBOARD_BAR_H + total_key_h + KEYBOARD_FOOTER_GAP
            max_letters = max(len(r) for r in KEYBOARD_ROWS_QWERTY)
            max_row_w = max(max_letters, len(KEYBOARD_ROW_NUMBERS)) * (KEY_W + KEY_GAP) + KEY_GAP
            special_w = KEY_W * 2
            special_row_w = len(KEYBOARD_SPECIAL) * (special_w + KEY_GAP) + KEY_GAP
            panel_w = max(max_row_w, special_row_w, int(220 * KEY_SCALE))
            row_y = dock_y + inset_y
            px = dock_x - MODAL_GAP_WEST - panel_w
            total_w = panel_w + MODAL_CONNECTOR_WIDTH
            return ("search", row_y, row_h, px, total_w, panel_h)
        if button_state.gallery_filter_modal_open:
            row_y = dock_y + inset_y + row_h + div
            px = dock_x - MODAL_GAP_WEST - MODAL_PANEL_W
            total_w = MODAL_PANEL_W + MODAL_CONNECTOR_WIDTH
            return ("filter", row_y, row_h, px, total_w, MODAL_PANEL_H)
        if button_state.gallery_sort_modal_open:
            row_y = dock_y + inset_y + 2 * (row_h + div)
            panel_h = MODAL_TITLE_H + len(SORT_OPTIONS) * MODAL_OPTION_H + 10
            px = dock_x - MODAL_GAP_WEST - MODAL_PANEL_W
            total_w = MODAL_PANEL_W + MODAL_CONNECTOR_WIDTH
            return ("sort", row_y, row_h, px, total_w, panel_h)
    return None


def _draw_combined_modal_strip(
    frame: np.ndarray,
    left: int,
    top: int,
    width: int,
    height: int,
    white_border: bool = False,
    bottom_bgr: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Draw one continuous blue gradient over button+modal strip. Border drawn separately (P-shape).
    If bottom_bgr is set, gradient runs BLUE -> bottom_bgr (so strip joins seamlessly with panel)."""
    if width < 1 or height < 1:
        return
    roi = frame[top : top + height, left : left + width]
    end_bgr = bottom_bgr if bottom_bgr is not None else MENU_ACTIVE_BLUE_LIGHT
    grad = _vertical_gradient(height, width, MENU_ACTIVE_BLUE, end_bgr)
    roi[:] = grad


def _draw_modal_panel_grow_slice(
    frame: np.ndarray,
    px: int,
    py: int,
    content_w: int,
    h: int,
    vis_left: int,
    vis_width: int,
    top_bgr: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Draw only the right vis_width pixels of the modal panel (grow-from-button).
    If top_bgr is set, gradient runs top_bgr -> LIGHT (joins seamlessly with strip above)."""
    if vis_width <= 0 or h <= 0:
        return
    fh, fw = frame.shape[:2]
    total_w = content_w + MODAL_CONNECTOR_WIDTH
    x0 = max(0, vis_left)
    x1 = min(fw, vis_left + vis_width)
    y0 = max(0, py)
    y1 = min(fh, py + h)
    w_actual = x1 - x0
    h_actual = y1 - y0
    if w_actual <= 0 or h_actual <= 0:
        return
    roi = frame[y0:y1, x0:x1]
    start_bgr = top_bgr if top_bgr is not None else MENU_ACTIVE_BLUE
    grad_full = _vertical_gradient(h, total_w, start_bgr, MENU_ACTIVE_BLUE_LIGHT)
    gy0 = y0 - py
    gy1 = gy0 + h_actual
    gx0 = total_w - w_actual
    gx1 = total_w
    grad_slice = grad_full[gy0:gy1, gx0:gx1]
    if grad_slice.shape[0] != roi.shape[0] or grad_slice.shape[1] != roi.shape[1]:
        return
    roi[:] = grad_slice


def _draw_modal_p_shape_border(
    frame: np.ndarray,
    vis_left: int,
    row_y: int,
    row_h: int,
    modal_right: int,
    modal_h: int,
    dock_x: int,
    dock_w: int,
) -> None:
    """Draw one continuous white border around the P-shape (modal + button row).
    Use button right edge (dock_x + dock_w - inset) so the right vertical border of the button shows."""
    button_right = dock_x + dock_w - DOCK_TOP_INSET_X
    pts = [
        (vis_left, row_y),
        (button_right, row_y),
        (button_right, row_y + row_h),
        (modal_right, row_y + row_h),
        (modal_right, row_y + modal_h),
        (vis_left, row_y + modal_h),
        (vis_left, row_y),
    ]
    pts_np = np.array(pts, dtype=np.int32)
    cv2.polylines(frame, [pts_np], isClosed=True, color=DOCK_ROW_WHITE_BORDER, thickness=1, lineType=cv2.LINE_AA)


def _modal_seam_color(row_h: int, panel_h: int) -> Tuple[int, int, int]:
    """Color at the seam between strip (row) and panel so gradient is continuous."""
    total_h = row_h + panel_h
    t = row_h / total_h if total_h > 0 else 1.0
    return (
        int(MENU_ACTIVE_BLUE[0] * (1 - t) + MENU_ACTIVE_BLUE_LIGHT[0] * t),
        int(MENU_ACTIVE_BLUE[1] * (1 - t) + MENU_ACTIVE_BLUE_LIGHT[1] * t),
        int(MENU_ACTIVE_BLUE[2] * (1 - t) + MENU_ACTIVE_BLUE_LIGHT[2] * t),
    )


def _draw_modal_panel_connected(
    frame: np.ndarray, px: int, py: int, content_w: int, h: int,
    top_bgr: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Draw modal panel with gradient; extend right by MODAL_CONNECTOR_WIDTH so blue is continuous
    with the dock row. If top_bgr is set, gradient runs top_bgr->LIGHT (seamless with strip above)."""
    fh, fw = frame.shape[:2]
    total_w = content_w + MODAL_CONNECTOR_WIDTH
    x0 = max(0, px)
    x1 = min(fw, px + total_w)
    y0 = max(0, py)
    y1 = min(fh, py + h)
    vis_w = x1 - x0
    vis_h = y1 - y0
    if vis_w <= 0 or vis_h <= 0:
        return
    roi = frame[y0:y1, x0:x1]
    start_bgr = top_bgr if top_bgr is not None else MENU_ACTIVE_BLUE
    grad_full = _vertical_gradient(h, total_w, start_bgr, MENU_ACTIVE_BLUE_LIGHT)
    # Visible slice of gradient (same region that falls in frame)
    gx0 = x0 - px
    gx1 = gx0 + vis_w
    gy0 = y0 - py
    gy1 = gy0 + vis_h
    grad_slice = grad_full[gy0:gy1, gx0:gx1]
    roi[:] = grad_slice
    # Left, top, bottom borders only (no right, so it flows into the row); clip to visible
    if x0 < x1:
        cv2.line(frame, (x0, y0), (x0, y1 - 1), DOCK_ROW_WHITE_BORDER, 1, cv2.LINE_AA)
    if y0 < y1:
        cv2.line(frame, (x0, y0), (x1 - 1, y0), DOCK_ROW_WHITE_BORDER, 1, cv2.LINE_AA)
        cv2.line(frame, (x0, y0), (x1 - 1, y0), DOCK_ROW_TOP_HIGHLIGHT, 1, cv2.LINE_AA)
        cv2.line(frame, (x0, y1 - 1), (x1 - 1, y1 - 1), DOCK_ROW_WHITE_BORDER, 1, cv2.LINE_AA)


def _draw_filter_modal(
    frame: np.ndarray, dock_x: int, dock_y: int, vis_left: int = -1, draw_panel: bool = True,
    content_visible: bool = True,
) -> None:
    """Draw filter-by-type panel aligned with top of Filter button; no screen dim."""
    fh, fw = frame.shape[:2]
    row_h = DOCK_ROW_HEIGHT
    filter_row_top_y = dock_y + DOCK_TOP_INSET_Y + row_h + DOCK_DIVIDER_THICKNESS
    px = dock_x - MODAL_GAP_WEST - MODAL_PANEL_W
    py = filter_row_top_y
    py = max(10, min(py, fh - MODAL_PANEL_H - 10))
    if draw_panel and content_visible and vis_left < 0:
        _draw_modal_panel_connected(
            frame, px, py, MODAL_PANEL_W, MODAL_PANEL_H,
            top_bgr=_modal_seam_color(DOCK_ROW_HEIGHT, MODAL_PANEL_H),
        )
    font = cv2.FONT_HERSHEY_SIMPLEX
    if content_visible and (vis_left < 0 or px + MODAL_PANEL_W >= vis_left):
        cv2.putText(frame, "Filter by type", (px + (MODAL_PANEL_W - 180) // 2, py + 32),
                    font, 0.7, SEARCH_BAR_TEXT_COLOR, 1, cv2.LINE_AA)
    for i, (label, value) in enumerate(FILTER_OPTIONS):
        oy = py + MODAL_TITLE_H + i * MODAL_OPTION_H
        btn_w, btn_h = MODAL_PANEL_W - 24, MODAL_OPTION_H - 8
        ox = px + (MODAL_PANEL_W - btn_w) // 2
        if vis_left >= 0 and ox + btn_w < vis_left:
            pass  # skip draw and register when clipped
        else:
            if content_visible:
                active = button_state.gallery_filter_type == value
                if active:
                    cv2.rectangle(frame, (ox, oy), (ox + btn_w, oy + btn_h), MODAL_ACTIVE_GOLD, -1)
                cv2.rectangle(frame, (ox, oy), (ox + btn_w, oy + btn_h), DOCK_ROW_WHITE_BORDER, 1, cv2.LINE_AA)
                (tw, _), _ = cv2.getTextSize(label, font, 0.55, 1)
                text_color = (0, 0, 0) if active else SEARCH_BAR_TEXT_COLOR
                cv2.putText(frame, label, (ox + (btn_w - tw) // 2, oy + btn_h // 2 + 6),
                            font, 0.55, text_color, 1, cv2.LINE_AA)
        key = f"gallery_filter_opt_{value}"
        if key not in menu_buttons:
            menu_buttons[key] = Button(ox, oy, btn_w, btn_h, label)
        else:
            menu_buttons[key].x, menu_buttons[key].y = ox, oy
            menu_buttons[key].w, menu_buttons[key].h = btn_w, btn_h
    total_w = MODAL_PANEL_W + MODAL_CONNECTOR_WIDTH
    if "gallery_filter_modal_panel" not in menu_buttons:
        menu_buttons["gallery_filter_modal_panel"] = Button(px, py, total_w, MODAL_PANEL_H, "")
    else:
        menu_buttons["gallery_filter_modal_panel"].x, menu_buttons["gallery_filter_modal_panel"].y = px, py
        menu_buttons["gallery_filter_modal_panel"].w, menu_buttons["gallery_filter_modal_panel"].h = total_w, MODAL_PANEL_H


def _draw_sort_modal(
    frame: np.ndarray, dock_x: int, dock_y: int, vis_left: int = -1, draw_panel: bool = True,
    content_visible: bool = True,
) -> None:
    """Draw sort panel aligned with top of Sort button; height scales with number of options."""
    fh, fw = frame.shape[:2]
    row_h = DOCK_ROW_HEIGHT
    sort_row_top_y = dock_y + DOCK_TOP_INSET_Y + 2 * (row_h + DOCK_DIVIDER_THICKNESS)
    panel_h = MODAL_TITLE_H + len(SORT_OPTIONS) * MODAL_OPTION_H + 10
    px = dock_x - MODAL_GAP_WEST - MODAL_PANEL_W
    py = sort_row_top_y
    py = max(10, min(py, fh - panel_h - 10))
    if draw_panel and content_visible and vis_left < 0:
        _draw_modal_panel_connected(
            frame, px, py, MODAL_PANEL_W, panel_h,
            top_bgr=_modal_seam_color(row_h, panel_h),
        )
    font = cv2.FONT_HERSHEY_SIMPLEX
    if content_visible and (vis_left < 0 or px + MODAL_PANEL_W >= vis_left):
        cv2.putText(frame, "Sort by", (px + (MODAL_PANEL_W - 80) // 2, py + 32),
                    font, 0.7, SEARCH_BAR_TEXT_COLOR, 1, cv2.LINE_AA)
    for i, (label, value) in enumerate(SORT_OPTIONS):
        oy = py + MODAL_TITLE_H + i * MODAL_OPTION_H
        btn_w, btn_h = MODAL_PANEL_W - 24, MODAL_OPTION_H - 8
        ox = px + (MODAL_PANEL_W - btn_w) // 2
        if vis_left >= 0 and ox + btn_w < vis_left:
            pass
        else:
            if content_visible:
                active = button_state.gallery_sort_by == value
                if active:
                    cv2.rectangle(frame, (ox, oy), (ox + btn_w, oy + btn_h), MODAL_ACTIVE_GOLD, -1)
                cv2.rectangle(frame, (ox, oy), (ox + btn_w, oy + btn_h), DOCK_ROW_WHITE_BORDER, 1, cv2.LINE_AA)
                (tw, _), _ = cv2.getTextSize(label, font, 0.5, 1)
                text_color = (0, 0, 0) if active else SEARCH_BAR_TEXT_COLOR
                cv2.putText(frame, label, (ox + (btn_w - tw) // 2, oy + btn_h // 2 + 6),
                            font, 0.5, text_color, 1, cv2.LINE_AA)
        key = f"gallery_sort_opt_{value}"
        if key not in menu_buttons:
            menu_buttons[key] = Button(ox, oy, btn_w, btn_h, label)
        else:
            menu_buttons[key].x, menu_buttons[key].y = ox, oy
            menu_buttons[key].w, menu_buttons[key].h = btn_w, btn_h
    total_w = MODAL_PANEL_W + MODAL_CONNECTOR_WIDTH
    if "gallery_sort_modal_panel" not in menu_buttons:
        menu_buttons["gallery_sort_modal_panel"] = Button(px, py, total_w, panel_h, "")
    else:
        menu_buttons["gallery_sort_modal_panel"].x, menu_buttons["gallery_sort_modal_panel"].y = px, py
        menu_buttons["gallery_sort_modal_panel"].w, menu_buttons["gallery_sort_modal_panel"].h = total_w, panel_h


def _draw_priority_modal(
    frame: np.ndarray, dock_x: int, dock_y: int, vis_left: int = -1, draw_panel: bool = True,
    content_visible: bool = True,
) -> None:
    """Priority picker aligned with Row 2 (Priority button) in select mode."""
    fh, _ = frame.shape[:2]
    row_h = DOCK_ROW_HEIGHT
    # Row 2 top = after Row 1 + divider
    row2_top_y = dock_y + DOCK_TOP_INSET_Y + row_h + DOCK_DIVIDER_THICKNESS
    n_opts = len(PRIORITY_OPTIONS)
    panel_h = MODAL_TITLE_H + n_opts * MODAL_OPTION_H + 10
    px = dock_x - MODAL_GAP_WEST - MODAL_PANEL_W
    py = row2_top_y
    py = max(10, min(py, fh - panel_h - 10))
    if draw_panel and content_visible and vis_left < 0:
        _draw_modal_panel_connected(
            frame, px, py, MODAL_PANEL_W, panel_h,
            top_bgr=_modal_seam_color(row_h, panel_h),
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    if content_visible and (vis_left < 0 or px + MODAL_PANEL_W >= vis_left):
        cv2.putText(frame, "Set Priority", (px + (MODAL_PANEL_W - 130) // 2, py + 32),
                    font, 0.7, SEARCH_BAR_TEXT_COLOR, 1, cv2.LINE_AA)

    # Determine which priority value is held by ALL currently selected items.
    rects = getattr(button_state, 'gallery_thumbnail_rects', [])
    sel_names = [r['filepath'].name for r in rects if r['idx'] in button_state.gallery_selected_items]
    priorities_map = getattr(button_state, 'gallery_file_priorities', {})
    selected_priorities: set = set()
    for fname in sel_names:
        pval = priorities_map.get(fname, "")
        if pval:
            selected_priorities.add(pval)

    for i, (label, value) in enumerate(PRIORITY_OPTIONS):
        oy = py + MODAL_TITLE_H + i * MODAL_OPTION_H
        btn_w, btn_h = MODAL_PANEL_W - 24, MODAL_OPTION_H - 8
        ox = px + (MODAL_PANEL_W - btn_w) // 2
        if vis_left >= 0 and ox + btn_w < vis_left:
            pass
        else:
            if content_visible:
                active = value in selected_priorities
                if active:
                    cv2.rectangle(frame, (ox, oy), (ox + btn_w, oy + btn_h), MODAL_ACTIVE_GOLD, -1)
                cv2.rectangle(frame, (ox, oy), (ox + btn_w, oy + btn_h), DOCK_ROW_WHITE_BORDER, 1, cv2.LINE_AA)
                dot_color = PRIORITY_COLORS[value]
                dot_cx = ox + 18
                dot_cy = oy + btn_h // 2
                draw_priority_circle_neon(frame, dot_cx, dot_cy, 7, dot_color)
                text_color = (0, 0, 0) if active else SEARCH_BAR_TEXT_COLOR
                (tw, _), _ = cv2.getTextSize(label, font, 0.55, 1)
                cv2.putText(frame, label, (ox + 34, oy + btn_h // 2 + 6),
                            font, 0.55, text_color, 1, cv2.LINE_AA)

        key = f"gallery_priority_opt_{value}"
        if key not in menu_buttons:
            menu_buttons[key] = Button(ox, oy, btn_w, btn_h, label)
        else:
            menu_buttons[key].x, menu_buttons[key].y = ox, oy
            menu_buttons[key].w, menu_buttons[key].h = btn_w, btn_h

    total_w = MODAL_PANEL_W + MODAL_CONNECTOR_WIDTH
    if "gallery_priority_modal_panel" not in menu_buttons:
        menu_buttons["gallery_priority_modal_panel"] = Button(px, py, total_w, panel_h, "")
    else:
        menu_buttons["gallery_priority_modal_panel"].x, menu_buttons["gallery_priority_modal_panel"].y = px, py
        menu_buttons["gallery_priority_modal_panel"].w, menu_buttons["gallery_priority_modal_panel"].h = total_w, panel_h


def _draw_tags_modal(
    frame: np.ndarray, dock_x: int, dock_y: int, vis_left: int = -1, draw_panel: bool = True,
    content_visible: bool = True,
) -> None:
    """Tags panel aligned with Row 1 (Tags button) in select mode."""
    fh, _ = frame.shape[:2]
    row1_top_y = dock_y + DOCK_TOP_INSET_Y
    n_tags = len(PRESET_TAGS)
    panel_h = MODAL_TITLE_H + n_tags * MODAL_OPTION_H + 10
    px = dock_x - MODAL_GAP_WEST - MODAL_PANEL_W
    py = row1_top_y
    py = max(10, min(py, fh - panel_h - 10))
    if draw_panel and content_visible and vis_left < 0:
        _draw_modal_panel_connected(
            frame, px, py, MODAL_PANEL_W, panel_h,
            top_bgr=_modal_seam_color(DOCK_ROW_HEIGHT, panel_h),
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    if content_visible and (vis_left < 0 or px + MODAL_PANEL_W >= vis_left):
        cv2.putText(frame, "Tags", (px + (MODAL_PANEL_W - 45) // 2, py + 32),
                    font, 0.7, SEARCH_BAR_TEXT_COLOR, 1, cv2.LINE_AA)

    for i, tag in enumerate(PRESET_TAGS):
        oy = py + MODAL_TITLE_H + i * MODAL_OPTION_H
        btn_w, btn_h = MODAL_PANEL_W - 24, MODAL_OPTION_H - 8
        ox = px + (MODAL_PANEL_W - btn_w) // 2
        if vis_left >= 0 and ox + btn_w < vis_left:
            pass
        else:
            if content_visible:
                file_tags = getattr(button_state, 'gallery_file_tags', {})
                active = False
                if button_state.gallery_selected_items:
                    rects = getattr(button_state, 'gallery_thumbnail_rects', [])
                    sel_names = [r['filepath'].name for r in rects if r['idx'] in button_state.gallery_selected_items]
                    active = bool(sel_names) and all(tag in file_tags.get(n, []) for n in sel_names)
                if active:
                    cv2.rectangle(frame, (ox, oy), (ox + btn_w, oy + btn_h), MODAL_ACTIVE_GOLD, -1)
                cv2.rectangle(frame, (ox, oy), (ox + btn_w, oy + btn_h), DOCK_ROW_WHITE_BORDER, 1, cv2.LINE_AA)
                _draw_icon_tag(frame, ox + 18, oy + btn_h // 2, SEARCH_BAR_TEXT_COLOR if not active else (0, 0, 0))
                text_color = (0, 0, 0) if active else SEARCH_BAR_TEXT_COLOR
                cv2.putText(frame, tag, (ox + 34, oy + btn_h // 2 + 6),
                            font, 0.52, text_color, 1, cv2.LINE_AA)

        key = f"gallery_tag_opt_{tag.lower().replace('-', '_').replace(' ', '_')}"
        if key not in menu_buttons:
            menu_buttons[key] = Button(ox, oy, btn_w, btn_h, tag)
        else:
            menu_buttons[key].x, menu_buttons[key].y = ox, oy
            menu_buttons[key].w, menu_buttons[key].h = btn_w, btn_h

    total_w = MODAL_PANEL_W + MODAL_CONNECTOR_WIDTH
    if "gallery_tags_modal_panel" not in menu_buttons:
        menu_buttons["gallery_tags_modal_panel"] = Button(px, py, total_w, panel_h, "")
    else:
        menu_buttons["gallery_tags_modal_panel"].x, menu_buttons["gallery_tags_modal_panel"].y = px, py
        menu_buttons["gallery_tags_modal_panel"].w, menu_buttons["gallery_tags_modal_panel"].h = total_w, panel_h


def _draw_rename_keyboard(
    frame: np.ndarray, dock_x: int, dock_y: int, vis_left: int = -1, draw_panel: bool = True,
    content_visible: bool = True,
) -> None:
    """Rename keyboard aligned with Row 3 (Rename button) in select mode."""
    fh, fw = frame.shape[:2]
    num_letter_rows = len(KEYBOARD_ROWS_QWERTY)
    num_rows = num_letter_rows + 1 + 1
    total_key_h = num_rows * (KEY_H + KEY_GAP) + KEY_GAP
    max_letters = max(len(r) for r in KEYBOARD_ROWS_QWERTY)
    max_row_w = max(max_letters, len(KEYBOARD_ROW_NUMBERS)) * (KEY_W + KEY_GAP) + KEY_GAP
    special_w = KEY_W * 2
    special_row_w = len(KEYBOARD_SPECIAL) * (special_w + KEY_GAP) + KEY_GAP
    panel_w = max(max_row_w, special_row_w, int(220 * KEY_SCALE))
    panel_h = KEYBOARD_BAR_H + total_key_h + KEYBOARD_FOOTER_GAP

    row3_top_y = dock_y + DOCK_TOP_INSET_Y + 2 * (DOCK_ROW_HEIGHT + DOCK_DIVIDER_THICKNESS)
    py = row3_top_y
    if py + panel_h > fh - 2:
        py = fh - panel_h - 2
    px = dock_x - MODAL_GAP_WEST - panel_w
    if px < 10:
        px = 10
    if draw_panel and content_visible and vis_left < 0:
        _draw_modal_panel_connected(
            frame, px, py, panel_w, panel_h,
            top_bgr=_modal_seam_color(DOCK_ROW_HEIGHT, panel_h),
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    if content_visible and (vis_left < 0 or px + panel_w >= vis_left):
        display = button_state.gallery_rename_query if button_state.gallery_rename_query else "New name..."
        cv2.putText(frame, display[:40], (px + 10, py + KEYBOARD_BAR_H - 10),
                    font, KEYBOARD_FONT_BAR, SEARCH_BAR_TEXT_COLOR, 1, cv2.LINE_AA)

    key_y = py + KEYBOARD_BAR_H + KEY_GAP
    for row in KEYBOARD_ROWS_QWERTY:
        key_x = px + (panel_w - (len(row) * (KEY_W + KEY_GAP) - KEY_GAP)) // 2
        for c in row:
            if vis_left >= 0 and key_x + KEY_W < vis_left:
                key_x += KEY_W + KEY_GAP
                continue
            if content_visible:
                draw_key_bg_clipped(frame, key_x, key_y, KEY_W, KEY_H)
                cv2.rectangle(frame, (key_x, key_y), (key_x + KEY_W, key_y + KEY_H), DOCK_ROW_WHITE_BORDER, 1, cv2.LINE_AA)
                (cw, ch), _ = cv2.getTextSize(c.upper(), font, KEYBOARD_FONT_KEY, 1)
                cv2.putText(frame, c.upper(), (key_x + (KEY_W - cw) // 2, key_y + (KEY_H + ch) // 2),
                            font, KEYBOARD_FONT_KEY, KEYBOARD_KEY_TEXT, 1, cv2.LINE_AA)
            key = f"rename_key_{c}"
            if key not in menu_buttons:
                menu_buttons[key] = Button(key_x, key_y, KEY_W, KEY_H, c)
            else:
                menu_buttons[key].x, menu_buttons[key].y = key_x, key_y
            key_x += KEY_W + KEY_GAP
        key_y += KEY_H + KEY_GAP

    key_x = px + (panel_w - (len(KEYBOARD_ROW_NUMBERS) * (KEY_W + KEY_GAP) - KEY_GAP)) // 2
    for c in KEYBOARD_ROW_NUMBERS:
        if vis_left >= 0 and key_x + KEY_W < vis_left:
            key_x += KEY_W + KEY_GAP
            continue
        if content_visible:
            draw_key_bg_clipped(frame, key_x, key_y, KEY_W, KEY_H)
            cv2.rectangle(frame, (key_x, key_y), (key_x + KEY_W, key_y + KEY_H), DOCK_ROW_WHITE_BORDER, 1, cv2.LINE_AA)
            (cw, ch), _ = cv2.getTextSize(c, font, KEYBOARD_FONT_KEY, 1)
            cv2.putText(frame, c, (key_x + (KEY_W - cw) // 2, key_y + (KEY_H + ch) // 2),
                        font, KEYBOARD_FONT_KEY, KEYBOARD_KEY_TEXT, 1, cv2.LINE_AA)
        key = f"rename_key_{c}"
        if key not in menu_buttons:
            menu_buttons[key] = Button(key_x, key_y, KEY_W, KEY_H, c)
        else:
            menu_buttons[key].x, menu_buttons[key].y = key_x, key_y
        key_x += KEY_W + KEY_GAP
    key_y += KEY_H + KEY_GAP

    key_x = px + (panel_w - (len(KEYBOARD_SPECIAL) * (special_w + KEY_GAP) - KEY_GAP)) // 2
    for label, value in KEYBOARD_SPECIAL:
        if vis_left >= 0 and key_x + special_w < vis_left:
            key_x += special_w + KEY_GAP
            continue
        if content_visible:
            draw_key_bg_clipped(frame, key_x, key_y, special_w, KEY_H)
            cv2.rectangle(frame, (key_x, key_y), (key_x + special_w, key_y + KEY_H), DOCK_ROW_WHITE_BORDER, 1, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(label, font, KEYBOARD_FONT_SPECIAL, 1)
            cv2.putText(frame, label, (key_x + (special_w - tw) // 2, key_y + KEY_H - 10),
                        font, KEYBOARD_FONT_SPECIAL, KEYBOARD_KEY_TEXT, 1, cv2.LINE_AA)
        key = f"rename_key_{value}"
        if key not in menu_buttons:
            menu_buttons[key] = Button(key_x, key_y, special_w, KEY_H, label)
        else:
            menu_buttons[key].x, menu_buttons[key].y = key_x, key_y
            menu_buttons[key].w = special_w
        key_x += special_w + KEY_GAP

    total_w = panel_w + MODAL_CONNECTOR_WIDTH
    if "rename_keyboard_panel" not in menu_buttons:
        menu_buttons["rename_keyboard_panel"] = Button(px, py, total_w, panel_h, "")
    else:
        menu_buttons["rename_keyboard_panel"].x, menu_buttons["rename_keyboard_panel"].y = px, py
        menu_buttons["rename_keyboard_panel"].w, menu_buttons["rename_keyboard_panel"].h = total_w, panel_h


def _draw_search_keyboard(
    frame: np.ndarray, dock_x: int, dock_y: int, vis_left: int = -1, draw_panel: bool = True,
    content_visible: bool = True,
) -> None:
    """Draw on-screen keyboard aligned with top of Search button (like Filter/Sort modals)."""
    fh, fw = frame.shape[:2]
    num_letter_rows = len(KEYBOARD_ROWS_QWERTY)
    num_rows = num_letter_rows + 1 + 1  # letters + number row + special
    total_key_h = num_rows * (KEY_H + KEY_GAP) + KEY_GAP
    max_letters = max(len(r) for r in KEYBOARD_ROWS_QWERTY)
    max_row_w = max(max_letters, len(KEYBOARD_ROW_NUMBERS)) * (KEY_W + KEY_GAP) + KEY_GAP
    special_w = KEY_W * 2
    special_row_w = len(KEYBOARD_SPECIAL) * (special_w + KEY_GAP) + KEY_GAP
    panel_w = max(max_row_w, special_row_w, int(220 * KEY_SCALE))
    panel_h = KEYBOARD_BAR_H + total_key_h + KEYBOARD_FOOTER_GAP
    search_row_top = dock_y + DOCK_TOP_INSET_Y
    py = search_row_top
    if py + panel_h > fh - 2:
        py = fh - panel_h - 2
    px = dock_x - MODAL_GAP_WEST - panel_w
    if px < 10:
        px = 10
    if draw_panel and content_visible and vis_left < 0:
        _draw_modal_panel_connected(
            frame, px, py, panel_w, panel_h,
            top_bgr=_modal_seam_color(DOCK_ROW_HEIGHT, panel_h),
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    if content_visible and (vis_left < 0 or px + panel_w >= vis_left):
        display = button_state.gallery_search_query if button_state.gallery_search_query else "Search..."
        cv2.putText(frame, display[:40], (px + 10, py + KEYBOARD_BAR_H - 10),
                    font, KEYBOARD_FONT_BAR, SEARCH_BAR_TEXT_COLOR, 1, cv2.LINE_AA)
    key_y = py + KEYBOARD_BAR_H + KEY_GAP
    for row in KEYBOARD_ROWS_QWERTY:
        key_x = px + (panel_w - (len(row) * (KEY_W + KEY_GAP) - KEY_GAP)) // 2
        for c in row:
            if vis_left >= 0 and key_x + KEY_W < vis_left:
                key_x += KEY_W + KEY_GAP
                continue
            if content_visible:
                draw_key_bg_clipped(frame, key_x, key_y, KEY_W, KEY_H)
                cv2.rectangle(frame, (key_x, key_y), (key_x + KEY_W, key_y + KEY_H), DOCK_ROW_WHITE_BORDER, 1, cv2.LINE_AA)
                (cw, ch), _ = cv2.getTextSize(c.upper(), font, KEYBOARD_FONT_KEY, 1)
                cv2.putText(frame, c.upper(), (key_x + (KEY_W - cw) // 2, key_y + (KEY_H + ch) // 2),
                            font, KEYBOARD_FONT_KEY, KEYBOARD_KEY_TEXT, 1, cv2.LINE_AA)
            key = f"search_key_{c}"
            if key not in menu_buttons:
                menu_buttons[key] = Button(key_x, key_y, KEY_W, KEY_H, c)
            else:
                menu_buttons[key].x, menu_buttons[key].y = key_x, key_y
            key_x += KEY_W + KEY_GAP
        key_y += KEY_H + KEY_GAP
    key_x = px + (panel_w - (len(KEYBOARD_ROW_NUMBERS) * (KEY_W + KEY_GAP) - KEY_GAP)) // 2
    for c in KEYBOARD_ROW_NUMBERS:
        if vis_left >= 0 and key_x + KEY_W < vis_left:
            key_x += KEY_W + KEY_GAP
            continue
        if content_visible:
            draw_key_bg_clipped(frame, key_x, key_y, KEY_W, KEY_H)
            cv2.rectangle(frame, (key_x, key_y), (key_x + KEY_W, key_y + KEY_H), DOCK_ROW_WHITE_BORDER, 1, cv2.LINE_AA)
            (cw, ch), _ = cv2.getTextSize(c, font, KEYBOARD_FONT_KEY, 1)
            cv2.putText(frame, c, (key_x + (KEY_W - cw) // 2, key_y + (KEY_H + ch) // 2),
                        font, KEYBOARD_FONT_KEY, KEYBOARD_KEY_TEXT, 1, cv2.LINE_AA)
        key = f"search_key_{c}"
        if key not in menu_buttons:
            menu_buttons[key] = Button(key_x, key_y, KEY_W, KEY_H, c)
        else:
            menu_buttons[key].x, menu_buttons[key].y = key_x, key_y
        key_x += KEY_W + KEY_GAP
    key_y += KEY_H + KEY_GAP
    key_x = px + (panel_w - (len(KEYBOARD_SPECIAL) * (special_w + KEY_GAP) - KEY_GAP)) // 2
    for label, value in KEYBOARD_SPECIAL:
        if vis_left >= 0 and key_x + special_w < vis_left:
            key_x += special_w + KEY_GAP
            continue
        if content_visible:
            draw_key_bg_clipped(frame, key_x, key_y, special_w, KEY_H)
            cv2.rectangle(frame, (key_x, key_y), (key_x + special_w, key_y + KEY_H), DOCK_ROW_WHITE_BORDER, 1, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(label, font, KEYBOARD_FONT_SPECIAL, 1)
            cv2.putText(frame, label, (key_x + (special_w - tw) // 2, key_y + KEY_H - 10),
                        font, KEYBOARD_FONT_SPECIAL, KEYBOARD_KEY_TEXT, 1, cv2.LINE_AA)
        key = f"search_key_{value}"
        if key not in menu_buttons:
            menu_buttons[key] = Button(key_x, key_y, special_w, KEY_H, label)
        else:
            menu_buttons[key].x, menu_buttons[key].y = key_x, key_y
            menu_buttons[key].w = special_w
        key_x += special_w + KEY_GAP
    total_w = panel_w + MODAL_CONNECTOR_WIDTH
    if "search_keyboard_panel" not in menu_buttons:
        menu_buttons["search_keyboard_panel"] = Button(px, py, total_w, panel_h, "")
    else:
        menu_buttons["search_keyboard_panel"].x, menu_buttons["search_keyboard_panel"].y = px, py
        menu_buttons["search_keyboard_panel"].w, menu_buttons["search_keyboard_panel"].h = total_w, panel_h


def _format_size(size_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def draw_storage_bar(
    frame: np.ndarray,
    dock_x: int,
    dock_w: int,
    header_h: int,
    items: List[Tuple[Path, str, datetime]],
    output_dir: Optional[Path],
) -> None:
    """
    Draw the storage bar inside the dock. Position is viewport-fixed so it
    stays visible (floats with you) as the grid scrolls. Uses live disk usage
    so the bar updates immediately after deletes.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    total_media_size = 0
    if items:
        try:
            total_media_size = sum(p.stat().st_size for p, _, _ in items)
        except OSError:
            pass

    if output_dir and output_dir.exists():
        try:
            path_str = os.path.realpath(str(output_dir))
            # After a delete, sync so filesystem reports updated usage (e.g. on SD card)
            if button_state.gallery_storage_dirty:
                os.sync()
                button_state.gallery_storage_dirty = False
            disk_usage = shutil.disk_usage(path_str)
            total_space = disk_usage.total
            used_space = disk_usage.used
        except OSError:
            total_space = 128 * 1024 * 1024 * 1024
            used_space = total_media_size
    else:
        if not items:
            return
        total_space = 128 * 1024 * 1024 * 1024
        used_space = total_media_size

    frame_h = frame.shape[0]
    bar_w = BAR_WIDTH_PX
    # Center storage block between Sort button bottom and frame bottom
    dock_y = header_h + 3
    sort_bottom = dock_y + 3 * DOCK_ROW_HEIGHT + 2 * DOCK_DIVIDER_THICKNESS
    storage_block_h = 6 + BAR_HEIGHT  # label gap + bar
    available_h = frame_h - sort_bottom
    block_top = sort_bottom + max(0, (available_h - storage_block_h) // 2)
    bar_y = block_top + 6

    used_percent = (used_space / total_space * 100) if total_space > 0 else 0
    free_space = total_space - used_space

    text_left = dock_x + 4

    # Compute text strings and measure max width (for bar centering)
    used_pct_str = f"{used_percent:.1f}%" if used_percent >= 0.1 else f"{used_percent:.2f}%"
    free_pct = 100.0 - used_percent
    free_pct_str = f"{free_pct:.1f}%" if free_pct >= 0.1 else f"{free_pct:.2f}%"
    used_size_str = _format_size(used_space)
    free_size_str = _format_size(free_space)

    font = cv2.FONT_HERSHEY_SIMPLEX
    max_text_w = 0
    for s in ("Free", "Used", free_pct_str, used_pct_str, free_size_str, used_size_str):
        (w, _), _ = cv2.getTextSize(s, font, 0.46, 1)
        max_text_w = max(max_text_w, w)
    text_right = text_left + max_text_w + BAR_TEXT_GAP_PX
    dock_right = dock_x + dock_w - BAR_RIGHT_MARGIN_PX
    bar_center = (text_right + dock_right) / 2
    bar_x = int(bar_center - bar_w / 2) - 8  # shift slightly left

    label_text = "STORAGE"
    label_scale = 0.52
    (label_w, _), _ = cv2.getTextSize(label_text, font, label_scale, 1)
    label_x = dock_x + (dock_w - label_w) // 2
    label_y = bar_y - 6
    cv2.putText(
        frame, label_text, (label_x, label_y),
        font, label_scale, (180, 180, 180), 1, cv2.LINE_AA
    )

    # Vertical bar (centered between text and dock right edge)
    cv2.rectangle(
        frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + BAR_HEIGHT),
        (45, 45, 50), -1
    )
    cv2.rectangle(
        frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + BAR_HEIGHT),
        (70, 70, 78), 2, cv2.LINE_AA
    )

    fill_x0 = bar_x + BAR_INSET
    fill_x1 = bar_x + bar_w - BAR_INSET
    fill_y0 = bar_y + BAR_INSET
    fill_y1 = bar_y + BAR_HEIGHT - BAR_INSET
    fill_area_h = fill_y1 - fill_y0

    filled_h = int(fill_area_h * min(used_percent / 100.0, 1.0))
    if used_space > 0 and filled_h < 2:
        filled_h = 2
    used_top = fill_y1 - filled_h
    used_color_top = MENU_ACTIVE_BLUE
    used_color_bot = MENU_ACTIVE_BLUE_LIGHT
    free_color_top = (115, 115, 120)
    free_color_bot = (145, 145, 150)
    if filled_h > 0:
        used_h = fill_y1 - used_top
        used_w = fill_x1 - fill_x0
        used_grad = np.zeros((used_h, used_w, 3), dtype=np.uint8)
        for c in range(3):
            used_grad[:, :, c] = np.linspace(
                used_color_top[c], used_color_bot[c], used_h, dtype=np.uint8
            ).reshape(-1, 1)
        frame[used_top:fill_y1, fill_x0:fill_x1] = used_grad
        free_h = used_top - fill_y0
        if free_h > 0:
            free_grad = np.zeros((free_h, used_w, 3), dtype=np.uint8)
            for c in range(3):
                free_grad[:, :, c] = np.linspace(
                    free_color_top[c], free_color_bot[c], free_h, dtype=np.uint8
                ).reshape(-1, 1)
            frame[fill_y0:used_top, fill_x0:fill_x1] = free_grad
        ridge_color = (90, 100, 120)  # blue-tinted ridge
        cv2.line(
            frame, (fill_x0, used_top), (fill_x1, used_top),
            ridge_color, 2, cv2.LINE_AA
        )

    # Text outside bar: label, percentage, size on separate lines
    # Vertically center the Free/Used block between STORAGE and bottom of bar
    percent_scale = 0.46
    text_color_used = MENU_ACTIVE_BLUE_LIGHT  # blue to match buttons
    text_color_free = (230, 230, 230)
    label_color = (160, 160, 165)
    line_h = 14
    gap_between_blocks = 4

    # Total height of text block: Free (3 lines) + gap + Used (3 lines)
    total_text_h = 3 * line_h + gap_between_blocks + 3 * line_h
    text_block_top = bar_y + (BAR_HEIGHT - total_text_h) // 2

    # Legend square size and gap
    LEGEND_SQ = 6
    LEGEND_GAP = 4
    legend_free_color = (140, 140, 145)  # grey (BGR)
    legend_used_color = MENU_ACTIVE_BLUE_LIGHT  # blue

    y = text_block_top
    sq_y = y - LEGEND_SQ - 2  # vertically center square with text
    cv2.rectangle(frame, (text_left, sq_y), (text_left + LEGEND_SQ, sq_y + LEGEND_SQ),
                  legend_free_color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (text_left, sq_y), (text_left + LEGEND_SQ, sq_y + LEGEND_SQ),
                  (180, 180, 185), 1, cv2.LINE_AA)
    cv2.putText(frame, "Free", (text_left + LEGEND_SQ + LEGEND_GAP, y), font, 0.45, label_color, 1, cv2.LINE_AA)
    y += line_h
    cv2.putText(frame, free_pct_str, (text_left, y), font, percent_scale, text_color_free, 1, cv2.LINE_AA)
    y += line_h
    cv2.putText(frame, free_size_str, (text_left, y), font, 0.42, text_color_free, 1, cv2.LINE_AA)

    y += line_h + gap_between_blocks
    sq_y = y - LEGEND_SQ - 2
    cv2.rectangle(frame, (text_left, sq_y), (text_left + LEGEND_SQ, sq_y + LEGEND_SQ),
                  legend_used_color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (text_left, sq_y), (text_left + LEGEND_SQ, sq_y + LEGEND_SQ),
                  (200, 200, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, "Used", (text_left + LEGEND_SQ + LEGEND_GAP, y), font, 0.45, text_color_used, 1, cv2.LINE_AA)
    y += line_h
    cv2.putText(frame, used_pct_str, (text_left, y), font, percent_scale, text_color_free, 1, cv2.LINE_AA)
    y += line_h
    cv2.putText(frame, used_size_str, (text_left, y), font, 0.42, text_color_free, 1, cv2.LINE_AA)


def draw_grid_side_dock(
    frame: np.ndarray,
    header_h: int,
    items: List[Tuple[Path, str, datetime]],
    output_dir: Optional[Path],
    scroll_offset: int = 0,
) -> None:
    """
    Draw the full-height side dock (delimited strip on the right) and the
    storage bar. The bar is viewport-fixed so it floats with you as you
    scroll the grid.
    """
    frame_h, frame_w = frame.shape[:2]
    dock_x = frame_w - GRID_SIDE_DOCK_WIDTH
    dock_y = header_h + 3
    dock_w = GRID_SIDE_DOCK_WIDTH
    dock_h = frame_h - dock_y

    # Dock background: grey-to-black gradient (same as viewer dock / top bar)
    dock_roi = frame[dock_y:dock_y + dock_h, dock_x:dock_x + dock_w]
    dock_roi[:] = _vertical_gradient(dock_h, dock_w, DOCK_GRADIENT_TOP, DOCK_GRADIENT_BOT)
    # Left edge: clear delimiter so it looks like a dock
    cv2.line(
        frame, (dock_x, dock_y), (dock_x, dock_y + dock_h),
        DOCK_EDGE_ACCENT, 2, cv2.LINE_AA
    )
    cv2.line(
        frame, (dock_x + 1, dock_y), (dock_x + 1, dock_y + dock_h),
        DOCK_EDGE, 1, cv2.LINE_AA
    )
    # Bottom edge: delimiter for bottom of side dock
    bottom_y = frame_h - 1
    cv2.line(
        frame, (dock_x, bottom_y), (dock_x + dock_w, bottom_y),
        DOCK_EDGE_ACCENT, 2, cv2.LINE_AA
    )
    cv2.line(
        frame, (dock_x, bottom_y - 1), (dock_x + dock_w, bottom_y - 1),
        DOCK_EDGE, 1, cv2.LINE_AA
    )

    # Modal slide animation: always run when opening/closing or when switching which modal is open
    info = _get_active_modal_info(frame, dock_x, dock_y, dock_w)
    any_open = info is not None
    active_key = info[0] if info else None
    target = 1.0 if any_open else 0.0
    if active_key != getattr(button_state, "gallery_modal_anim_active_key", ""):
        button_state.gallery_modal_anim_active_key = active_key or ""
        button_state.gallery_modal_anim_target = target
        button_state.gallery_modal_anim_start_t = time.time()
        button_state.gallery_modal_anim_progress_at_flip = (
            0.0 if any_open else button_state.gallery_modal_anim_progress
        )
        if any_open:
            button_state.gallery_modal_anim_progress = 0.0
    elif target != button_state.gallery_modal_anim_target:
        button_state.gallery_modal_anim_target = target
        button_state.gallery_modal_anim_start_t = time.time()
        button_state.gallery_modal_anim_progress_at_flip = button_state.gallery_modal_anim_progress
    now = time.time()
    elapsed = now - button_state.gallery_modal_anim_start_t
    t = min(1.0, elapsed / MODAL_ANIM_DURATION_S)
    if button_state.gallery_modal_anim_target >= 0.5:
        button_state.gallery_modal_anim_progress = button_state.gallery_modal_anim_progress_at_flip + (1.0 - button_state.gallery_modal_anim_progress_at_flip) * t
    else:
        button_state.gallery_modal_anim_progress = button_state.gallery_modal_anim_progress_at_flip * (1.0 - t)
    progress = button_state.gallery_modal_anim_progress
    # Ease-in-out (smoothstep) to smooth the grow animation and reduce flicker/secondary movement
    t = progress
    progress_eased = t * t * (3.0 - 2.0 * t) if 0 <= t <= 1 else t
    modal_px = modal_total_w = modal_h = row_y = row_h = 0
    if info is not None:
        _, row_y, row_h, modal_px, modal_total_w, modal_h = info

    # Grow-from-button: visible region is the right (progress * total_w) part of the modal
    vis_left = -1
    if info is not None and progress_eased > 0.001:
        vis_width = max(1, int(progress_eased * modal_total_w))
        vis_left = modal_px + modal_total_w - vis_width
        strip_right = dock_x + dock_w
        strip_width = max(1, strip_right - vis_left)
        # One continuous gradient: strip BLUE->mid, panel mid->LIGHT so no seam at row_h
        t_mid = row_h / modal_h if modal_h > 0 else 1.0
        mid_bgr: Tuple[int, int, int] = (
            int(MENU_ACTIVE_BLUE[0] * (1 - t_mid) + MENU_ACTIVE_BLUE_LIGHT[0] * t_mid),
            int(MENU_ACTIVE_BLUE[1] * (1 - t_mid) + MENU_ACTIVE_BLUE_LIGHT[1] * t_mid),
            int(MENU_ACTIVE_BLUE[2] * (1 - t_mid) + MENU_ACTIVE_BLUE_LIGHT[2] * t_mid),
        )
        _draw_combined_modal_strip(
            frame, vis_left, row_y, strip_width, row_h,
            white_border=False, bottom_bgr=mid_bgr,
        )
        if modal_h > row_h:
            content_w = modal_total_w - MODAL_CONNECTOR_WIDTH
            _draw_modal_panel_grow_slice(
                frame, modal_px, row_y + row_h, content_w, modal_h - row_h,
                vis_left, vis_width, top_bgr=mid_bgr,
            )

    # Draw modal content only after growth finishes to avoid content appearing before the panel
    content_visible = (info is not None and progress_eased >= 0.999)
    draw_panel = True
    if button_state.gallery_select_mode:
        if button_state.gallery_priority_modal_open:
            _draw_priority_modal(frame, dock_x, dock_y, vis_left if active_key == "priority" else -1, draw_panel, content_visible)
        if button_state.gallery_rename_modal_open:
            _draw_rename_keyboard(frame, dock_x, dock_y, vis_left if active_key == "rename" else -1, draw_panel, content_visible)
    else:
        if button_state.gallery_filter_modal_open:
            _draw_filter_modal(frame, dock_x, dock_y, vis_left if active_key == "filter" else -1, draw_panel, content_visible)
        if button_state.gallery_sort_modal_open:
            _draw_sort_modal(frame, dock_x, dock_y, vis_left if active_key == "sort" else -1, draw_panel, content_visible)
        if button_state.gallery_search_keyboard_open:
            _draw_search_keyboard(frame, dock_x, dock_y, vis_left if active_key == "search" else -1, draw_panel, content_visible)

    _draw_dock_top_rows(frame, dock_x, dock_y, dock_w, active_key, progress_eased)

    # P-shaped continuous outer border (button + modal); grows with the modal
    if info is not None and progress_eased > 0.001 and vis_left >= 0:
        modal_right = modal_px + modal_total_w
        _draw_modal_p_shape_border(frame, vis_left, row_y, row_h, modal_right, modal_h, dock_x, dock_w)

    draw_storage_bar(
        frame, dock_x, dock_w, header_h, items, output_dir
    )
