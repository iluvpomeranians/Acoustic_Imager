"""
Grid side dock: vertical strip on the right in gallery grid view.

Dock is delimited from the grid; the storage bar floats with the viewport
(stays visible) as you scroll up/down.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple
import shutil

import cv2
import numpy as np

from ..config import MENU_ACTIVE_BLUE, MENU_ACTIVE_BLUE_LIGHT, MODAL_ACTIVE_GOLD, DOCK_GRADIENT_TOP, DOCK_GRADIENT_BOT
from ..state import button_state
from .button import Button, menu_buttons
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
BAR_HEIGHT = 220
BAR_INSET = 2
# Bar is 15% skinnier than dock, centered (1px gap each side of dock)
BAR_WIDTH_RATIO = 0.85

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


def _draw_dock_row(
    frame: np.ndarray,
    x0: int,
    y0: int,
    w: int,
    h: int,
    label: str,
    icon_right: Optional[str] = None,
) -> None:
    """Draw one top-dock row: gradient fill, subtle border, icon above text, centered."""
    if w < 4 or h < 8:
        return
    roi = frame[y0:y0 + h, x0:x0 + w]
    gradient = _vertical_gradient(h, w, MENU_ACTIVE_BLUE, MENU_ACTIVE_BLUE_LIGHT)
    roi[:] = gradient
    cv2.rectangle(frame, (x0, y0), (x0 + w - 1, y0 + h - 1), DOCK_ROW_BORDER, 1, cv2.LINE_AA)
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


def _draw_dock_top_rows(frame: np.ndarray, dock_x: int, dock_y: int, dock_w: int) -> None:
    """Draw top dock rows.

    Normal mode  → Search / Filter / Sort
    Select mode  → Tags / Priority / Rename
    """
    if button_state.gallery_select_mode:
        _draw_select_mode_rows(frame, dock_x, dock_y, dock_w)
        return

    inset_x = DOCK_TOP_INSET_X
    inset_y = DOCK_TOP_INSET_Y
    row_h = DOCK_ROW_HEIGHT
    x0 = dock_x + inset_x
    w = dock_w - 2 * inset_x
    if w < 10:
        return
    y = dock_y + inset_y
    # Row 1: Search
    _draw_dock_row(frame, x0, y, w, row_h, SEARCH_BAR_PLACEHOLDER, icon_right="search")
    if "gallery_dock_search" not in menu_buttons:
        menu_buttons["gallery_dock_search"] = Button(x0, y, w, row_h, "Search")
    else:
        b = menu_buttons["gallery_dock_search"]
        b.x, b.y, b.w, b.h = x0, y, w, row_h
    y += row_h
    for _ in range(DOCK_DIVIDER_THICKNESS):
        cv2.line(frame, (dock_x, y), (dock_x + dock_w, y), DOCK_DIVIDER_COLOR, 1, cv2.LINE_AA)
        y += 1
    # Row 2: Filter
    _draw_dock_row(frame, x0, y, w, row_h, DOCK_ROW_FILTER_LABEL, icon_right="filter")
    if "gallery_dock_filter" not in menu_buttons:
        menu_buttons["gallery_dock_filter"] = Button(x0, y, w, row_h, "Filter")
    else:
        b = menu_buttons["gallery_dock_filter"]
        b.x, b.y, b.w, b.h = x0, y, w, row_h
    y += row_h
    for _ in range(DOCK_DIVIDER_THICKNESS):
        cv2.line(frame, (dock_x, y), (dock_x + dock_w, y), DOCK_DIVIDER_COLOR, 1, cv2.LINE_AA)
        y += 1
    # Row 3: Sort
    _draw_dock_row(frame, x0, y, w, row_h, DOCK_ROW_THIRD_LABEL, icon_right="sort")
    if "gallery_dock_sort" not in menu_buttons:
        menu_buttons["gallery_dock_sort"] = Button(x0, y, w, row_h, "Sort")
    else:
        b = menu_buttons["gallery_dock_sort"]
        b.x, b.y, b.w, b.h = x0, y, w, row_h


def _draw_select_mode_rows(frame: np.ndarray, dock_x: int, dock_y: int, dock_w: int) -> None:
    """Draw Tags / Priority / Rename rows shown when gallery_select_mode is active."""
    inset_x = DOCK_TOP_INSET_X
    inset_y = DOCK_TOP_INSET_Y
    row_h = DOCK_ROW_HEIGHT
    x0 = dock_x + inset_x
    w = dock_w - 2 * inset_x
    if w < 10:
        return
    y = dock_y + inset_y

    # Row 1: Tags
    _draw_dock_row(frame, x0, y, w, row_h, "Tags", icon_right="tag")
    if "gallery_dock_tags" not in menu_buttons:
        menu_buttons["gallery_dock_tags"] = Button(x0, y, w, row_h, "Tags")
    else:
        b = menu_buttons["gallery_dock_tags"]
        b.x, b.y, b.w, b.h = x0, y, w, row_h
    y += row_h
    for _ in range(DOCK_DIVIDER_THICKNESS):
        cv2.line(frame, (dock_x, y), (dock_x + dock_w, y), DOCK_DIVIDER_COLOR, 1, cv2.LINE_AA)
        y += 1

    # Row 2: Priority
    _draw_dock_row(frame, x0, y, w, row_h, "Priority", icon_right="priority")
    if "gallery_dock_priority" not in menu_buttons:
        menu_buttons["gallery_dock_priority"] = Button(x0, y, w, row_h, "Priority")
    else:
        b = menu_buttons["gallery_dock_priority"]
        b.x, b.y, b.w, b.h = x0, y, w, row_h
    y += row_h
    for _ in range(DOCK_DIVIDER_THICKNESS):
        cv2.line(frame, (dock_x, y), (dock_x + dock_w, y), DOCK_DIVIDER_COLOR, 1, cv2.LINE_AA)
        y += 1

    # Row 3: Rename
    _draw_dock_row(frame, x0, y, w, row_h, "Rename", icon_right="pen")
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
MODAL_OPTION_H = 44
MODAL_TITLE_H = 50

# Search & Rename keyboards: identical size; reduced header/footer so they line up with dock rows
KEYBOARD_ROWS_QWERTY = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
KEYBOARD_ROW_NUMBERS = "1234567890"
KEYBOARD_SPECIAL = [("Backspace", "backspace"), ("Clear", "clear"), ("Done", "done")]
KEY_SCALE = 1.875  # compact + 25% larger (1.5 * 1.25)
KEY_W, KEY_H = int(28 * KEY_SCALE), int(28 * KEY_SCALE)
KEY_GAP = int(4 * KEY_SCALE)
KEYBOARD_BAR_H = int(22 * KEY_SCALE)  # smaller header so Rename can stay aligned
KEYBOARD_FOOTER_GAP = 0               # minimal footer so it fits without shifting
KEYBOARD_FONT_BAR = 0.6
KEYBOARD_FONT_KEY = 0.55
KEYBOARD_FONT_SPECIAL = 0.5
KEYBOARD_MARGIN_BOTTOM = 24


# Modal as extension of dock: gap west of dock, no overlay.
# Connector width so modal blue meets the row (continuous from button to modal).
MODAL_GAP_WEST = 8
MODAL_CONNECTOR_WIDTH = MODAL_GAP_WEST + DOCK_TOP_INSET_X  # 10px: modal extends to row start


def _draw_modal_panel_connected(
    frame: np.ndarray, px: int, py: int, content_w: int, h: int
) -> None:
    """Draw modal panel with gradient; extend right by MODAL_CONNECTOR_WIDTH so blue is continuous
    with the dock row. No right border so it visually merges with the button."""
    total_w = content_w + MODAL_CONNECTOR_WIDTH
    roi = frame[py : py + h, px : px + total_w]
    grad = _vertical_gradient(h, total_w, MENU_ACTIVE_BLUE, MENU_ACTIVE_BLUE_LIGHT)
    roi[:] = grad
    # Left, top, bottom borders only (no right, so it flows into the row)
    cv2.line(frame, (px, py), (px, py + h - 1), DOCK_ROW_BORDER, 1, cv2.LINE_AA)
    cv2.line(frame, (px, py), (px + total_w, py), DOCK_ROW_BORDER, 1, cv2.LINE_AA)
    cv2.line(frame, (px, py + h - 1), (px + total_w, py + h - 1), DOCK_ROW_BORDER, 1, cv2.LINE_AA)
    cv2.line(frame, (px, py), (px + total_w, py), DOCK_ROW_TOP_HIGHLIGHT, 1, cv2.LINE_AA)


def _draw_filter_modal(frame: np.ndarray, dock_x: int, dock_y: int) -> None:
    """Draw filter-by-type panel aligned with top of Filter button; no screen dim."""
    fh, fw = frame.shape[:2]
    row_h = DOCK_ROW_HEIGHT
    filter_row_top_y = dock_y + DOCK_TOP_INSET_Y + row_h + DOCK_DIVIDER_THICKNESS
    px = dock_x - MODAL_GAP_WEST - MODAL_PANEL_W
    py = filter_row_top_y
    py = max(10, min(py, fh - MODAL_PANEL_H - 10))
    _draw_modal_panel_connected(frame, px, py, MODAL_PANEL_W, MODAL_PANEL_H)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Filter by type", (px + (MODAL_PANEL_W - 180) // 2, py + 32),
                font, 0.7, SEARCH_BAR_TEXT_COLOR, 1, cv2.LINE_AA)
    for i, (label, value) in enumerate(FILTER_OPTIONS):
        oy = py + MODAL_TITLE_H + i * MODAL_OPTION_H
        btn_w, btn_h = MODAL_PANEL_W - 24, MODAL_OPTION_H - 8
        ox = px + (MODAL_PANEL_W - btn_w) // 2
        active = button_state.gallery_filter_type == value
        if active:
            cv2.rectangle(frame, (ox, oy), (ox + btn_w, oy + btn_h), MODAL_ACTIVE_GOLD, -1)
        cv2.rectangle(frame, (ox, oy), (ox + btn_w, oy + btn_h), DOCK_ROW_BORDER, 1, cv2.LINE_AA)
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


def _draw_sort_modal(frame: np.ndarray, dock_x: int, dock_y: int) -> None:
    """Draw sort panel aligned with top of Sort button; height scales with number of options."""
    fh, fw = frame.shape[:2]
    row_h = DOCK_ROW_HEIGHT
    sort_row_top_y = dock_y + DOCK_TOP_INSET_Y + 2 * (row_h + DOCK_DIVIDER_THICKNESS)
    panel_h = MODAL_TITLE_H + len(SORT_OPTIONS) * MODAL_OPTION_H + 10
    px = dock_x - MODAL_GAP_WEST - MODAL_PANEL_W
    py = sort_row_top_y
    py = max(10, min(py, fh - panel_h - 10))
    _draw_modal_panel_connected(frame, px, py, MODAL_PANEL_W, panel_h)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Sort by", (px + (MODAL_PANEL_W - 80) // 2, py + 32),
                font, 0.7, SEARCH_BAR_TEXT_COLOR, 1, cv2.LINE_AA)
    for i, (label, value) in enumerate(SORT_OPTIONS):
        oy = py + MODAL_TITLE_H + i * MODAL_OPTION_H
        btn_w, btn_h = MODAL_PANEL_W - 24, MODAL_OPTION_H - 8
        ox = px + (MODAL_PANEL_W - btn_w) // 2
        active = button_state.gallery_sort_by == value
        if active:
            cv2.rectangle(frame, (ox, oy), (ox + btn_w, oy + btn_h), MODAL_ACTIVE_GOLD, -1)
        cv2.rectangle(frame, (ox, oy), (ox + btn_w, oy + btn_h), DOCK_ROW_BORDER, 1, cv2.LINE_AA)
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


def _draw_priority_modal(frame: np.ndarray, dock_x: int, dock_y: int) -> None:
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

    _draw_modal_panel_connected(frame, px, py, MODAL_PANEL_W, panel_h)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Set Priority", (px + (MODAL_PANEL_W - 130) // 2, py + 32),
                font, 0.7, SEARCH_BAR_TEXT_COLOR, 1, cv2.LINE_AA)

    # Determine which priority value is held by ALL currently selected items.
    # Use gallery_thumbnail_rects (populated each frame by gallery.py) so we
    # know the filenames of selected items without a circular import.
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
        active = value in selected_priorities
        if active:
            cv2.rectangle(frame, (ox, oy), (ox + btn_w, oy + btn_h), MODAL_ACTIVE_GOLD, -1)
        cv2.rectangle(frame, (ox, oy), (ox + btn_w, oy + btn_h), DOCK_ROW_BORDER, 1, cv2.LINE_AA)

        # Neon priority circle (same as gallery grid)
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


def _draw_tags_modal(frame: np.ndarray, dock_x: int, dock_y: int) -> None:
    """Tags panel aligned with Row 1 (Tags button) in select mode."""
    fh, _ = frame.shape[:2]
    row1_top_y = dock_y + DOCK_TOP_INSET_Y
    n_tags = len(PRESET_TAGS)
    panel_h = MODAL_TITLE_H + n_tags * MODAL_OPTION_H + 10
    px = dock_x - MODAL_GAP_WEST - MODAL_PANEL_W
    py = row1_top_y
    py = max(10, min(py, fh - panel_h - 10))

    _draw_modal_panel_connected(frame, px, py, MODAL_PANEL_W, panel_h)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Tags", (px + (MODAL_PANEL_W - 45) // 2, py + 32),
                font, 0.7, SEARCH_BAR_TEXT_COLOR, 1, cv2.LINE_AA)

    for i, tag in enumerate(PRESET_TAGS):
        oy = py + MODAL_TITLE_H + i * MODAL_OPTION_H
        btn_w, btn_h = MODAL_PANEL_W - 24, MODAL_OPTION_H - 8
        ox = px + (MODAL_PANEL_W - btn_w) // 2

        # A tag is "active" if ALL selected items have it.
        # Use gallery_thumbnail_rects (populated each frame by gallery.py) to avoid circular import.
        file_tags = getattr(button_state, 'gallery_file_tags', {})
        active = False
        if button_state.gallery_selected_items:
            rects = getattr(button_state, 'gallery_thumbnail_rects', [])
            sel_names = [r['filepath'].name for r in rects if r['idx'] in button_state.gallery_selected_items]
            active = bool(sel_names) and all(tag in file_tags.get(n, []) for n in sel_names)

        if active:
            cv2.rectangle(frame, (ox, oy), (ox + btn_w, oy + btn_h), MODAL_ACTIVE_GOLD, -1)
        cv2.rectangle(frame, (ox, oy), (ox + btn_w, oy + btn_h), DOCK_ROW_BORDER, 1, cv2.LINE_AA)

        # Small tag icon before label
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


def _draw_rename_keyboard(frame: np.ndarray, dock_x: int, dock_y: int) -> None:
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

    _draw_modal_panel_connected(frame, px, py, panel_w, panel_h)

    font = cv2.FONT_HERSHEY_SIMPLEX
    display = button_state.gallery_rename_query if button_state.gallery_rename_query else "New name..."
    cv2.putText(frame, display[:40], (px + 10, py + KEYBOARD_BAR_H - 10),
                font, KEYBOARD_FONT_BAR, SEARCH_BAR_TEXT_COLOR, 1, cv2.LINE_AA)

    key_y = py + KEYBOARD_BAR_H + KEY_GAP
    for row in KEYBOARD_ROWS_QWERTY:
        key_x = px + (panel_w - (len(row) * (KEY_W + KEY_GAP) - KEY_GAP)) // 2
        for c in row:
            key_roi = frame[key_y: key_y + KEY_H, key_x: key_x + KEY_W]
            key_roi[:] = _vertical_gradient(KEY_H, KEY_W, MENU_ACTIVE_BLUE, MENU_ACTIVE_BLUE_LIGHT)
            cv2.rectangle(frame, (key_x, key_y), (key_x + KEY_W, key_y + KEY_H), DOCK_ROW_BORDER, 1, cv2.LINE_AA)
            (cw, ch), _ = cv2.getTextSize(c.upper(), font, KEYBOARD_FONT_KEY, 1)
            cv2.putText(frame, c.upper(), (key_x + (KEY_W - cw) // 2, key_y + (KEY_H + ch) // 2),
                        font, KEYBOARD_FONT_KEY, SEARCH_BAR_TEXT_COLOR, 1, cv2.LINE_AA)
            key = f"rename_key_{c}"
            if key not in menu_buttons:
                menu_buttons[key] = Button(key_x, key_y, KEY_W, KEY_H, c)
            else:
                menu_buttons[key].x, menu_buttons[key].y = key_x, key_y
            key_x += KEY_W + KEY_GAP
        key_y += KEY_H + KEY_GAP

    key_x = px + (panel_w - (len(KEYBOARD_ROW_NUMBERS) * (KEY_W + KEY_GAP) - KEY_GAP)) // 2
    for c in KEYBOARD_ROW_NUMBERS:
        key_roi = frame[key_y: key_y + KEY_H, key_x: key_x + KEY_W]
        key_roi[:] = _vertical_gradient(KEY_H, KEY_W, MENU_ACTIVE_BLUE, MENU_ACTIVE_BLUE_LIGHT)
        cv2.rectangle(frame, (key_x, key_y), (key_x + KEY_W, key_y + KEY_H), DOCK_ROW_BORDER, 1, cv2.LINE_AA)
        (cw, ch), _ = cv2.getTextSize(c, font, KEYBOARD_FONT_KEY, 1)
        cv2.putText(frame, c, (key_x + (KEY_W - cw) // 2, key_y + (KEY_H + ch) // 2),
                    font, KEYBOARD_FONT_KEY, SEARCH_BAR_TEXT_COLOR, 1, cv2.LINE_AA)
        key = f"rename_key_{c}"
        if key not in menu_buttons:
            menu_buttons[key] = Button(key_x, key_y, KEY_W, KEY_H, c)
        else:
            menu_buttons[key].x, menu_buttons[key].y = key_x, key_y
        key_x += KEY_W + KEY_GAP
    key_y += KEY_H + KEY_GAP

    key_x = px + (panel_w - (len(KEYBOARD_SPECIAL) * (special_w + KEY_GAP) - KEY_GAP)) // 2
    for label, value in KEYBOARD_SPECIAL:
        key_roi = frame[key_y: key_y + KEY_H, key_x: key_x + special_w]
        key_roi[:] = _vertical_gradient(KEY_H, special_w, MENU_ACTIVE_BLUE, MENU_ACTIVE_BLUE_LIGHT)
        cv2.rectangle(frame, (key_x, key_y), (key_x + special_w, key_y + KEY_H), DOCK_ROW_BORDER, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(label, font, KEYBOARD_FONT_SPECIAL, 1)
        cv2.putText(frame, label, (key_x + (special_w - tw) // 2, key_y + KEY_H - 10),
                    font, KEYBOARD_FONT_SPECIAL, SEARCH_BAR_TEXT_COLOR, 1, cv2.LINE_AA)
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


def _draw_search_keyboard(frame: np.ndarray, dock_x: int, dock_y: int) -> None:
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

    _draw_modal_panel_connected(frame, px, py, panel_w, panel_h)

    font = cv2.FONT_HERSHEY_SIMPLEX
    display = button_state.gallery_search_query if button_state.gallery_search_query else "Search..."
    cv2.putText(frame, display[:40], (px + 10, py + KEYBOARD_BAR_H - 10),
                font, KEYBOARD_FONT_BAR, SEARCH_BAR_TEXT_COLOR, 1, cv2.LINE_AA)
    key_y = py + KEYBOARD_BAR_H + KEY_GAP
    # QWERTY letter rows
    for row in KEYBOARD_ROWS_QWERTY:
        key_x = px + (panel_w - (len(row) * (KEY_W + KEY_GAP) - KEY_GAP)) // 2
        for c in row:
            key_roi = frame[key_y : key_y + KEY_H, key_x : key_x + KEY_W]
            key_grad = _vertical_gradient(KEY_H, KEY_W, MENU_ACTIVE_BLUE, MENU_ACTIVE_BLUE_LIGHT)
            key_roi[:] = key_grad
            cv2.rectangle(frame, (key_x, key_y), (key_x + KEY_W, key_y + KEY_H), DOCK_ROW_BORDER, 1, cv2.LINE_AA)
            (cw, ch), _ = cv2.getTextSize(c.upper(), font, KEYBOARD_FONT_KEY, 1)
            cv2.putText(frame, c.upper(), (key_x + (KEY_W - cw) // 2, key_y + (KEY_H + ch) // 2),
                        font, KEYBOARD_FONT_KEY, SEARCH_BAR_TEXT_COLOR, 1, cv2.LINE_AA)
            key = f"search_key_{c}"
            if key not in menu_buttons:
                menu_buttons[key] = Button(key_x, key_y, KEY_W, KEY_H, c)
            else:
                menu_buttons[key].x, menu_buttons[key].y = key_x, key_y
            key_x += KEY_W + KEY_GAP
        key_y += KEY_H + KEY_GAP
    # Dedicated number row (1 2 3 4 5 6 7 8 9 0)
    key_x = px + (panel_w - (len(KEYBOARD_ROW_NUMBERS) * (KEY_W + KEY_GAP) - KEY_GAP)) // 2
    for c in KEYBOARD_ROW_NUMBERS:
        key_roi = frame[key_y : key_y + KEY_H, key_x : key_x + KEY_W]
        key_grad = _vertical_gradient(KEY_H, KEY_W, MENU_ACTIVE_BLUE, MENU_ACTIVE_BLUE_LIGHT)
        key_roi[:] = key_grad
        cv2.rectangle(frame, (key_x, key_y), (key_x + KEY_W, key_y + KEY_H), DOCK_ROW_BORDER, 1, cv2.LINE_AA)
        (cw, ch), _ = cv2.getTextSize(c, font, KEYBOARD_FONT_KEY, 1)
        cv2.putText(frame, c, (key_x + (KEY_W - cw) // 2, key_y + (KEY_H + ch) // 2),
                    font, KEYBOARD_FONT_KEY, SEARCH_BAR_TEXT_COLOR, 1, cv2.LINE_AA)
        key = f"search_key_{c}"
        if key not in menu_buttons:
            menu_buttons[key] = Button(key_x, key_y, KEY_W, KEY_H, c)
        else:
            menu_buttons[key].x, menu_buttons[key].y = key_x, key_y
        key_x += KEY_W + KEY_GAP
    key_y += KEY_H + KEY_GAP
    # Special row: Backspace, Clear, Done
    key_x = px + (panel_w - (len(KEYBOARD_SPECIAL) * (special_w + KEY_GAP) - KEY_GAP)) // 2
    for label, value in KEYBOARD_SPECIAL:
        key_roi = frame[key_y : key_y + KEY_H, key_x : key_x + special_w]
        key_grad = _vertical_gradient(KEY_H, special_w, MENU_ACTIVE_BLUE, MENU_ACTIVE_BLUE_LIGHT)
        key_roi[:] = key_grad
        cv2.rectangle(frame, (key_x, key_y), (key_x + special_w, key_y + KEY_H), DOCK_ROW_BORDER, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(label, font, KEYBOARD_FONT_SPECIAL, 1)
        cv2.putText(frame, label, (key_x + (special_w - tw) // 2, key_y + KEY_H - 10),
                    font, KEYBOARD_FONT_SPECIAL, SEARCH_BAR_TEXT_COLOR, 1, cv2.LINE_AA)
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
    max_w = dock_w - 2
    bar_w = max(20, int(max_w * BAR_WIDTH_RATIO))
    bar_x = dock_x + 1 + (max_w - bar_w) // 2
    bar_y = frame_h - BAR_BOTTOM_MARGIN_PX - BAR_HEIGHT

    label_text = "STORAGE"
    label_scale = 0.54
    (label_w, _), _ = cv2.getTextSize(label_text, font, label_scale, 1)
    label_x = dock_x + (dock_w - label_w) // 2
    label_y = bar_y - 8
    cv2.putText(
        frame, label_text, (label_x, label_y),
        font, label_scale, (180, 180, 180), 1, cv2.LINE_AA
    )

    used_percent = (used_space / total_space * 100) if total_space > 0 else 0
    free_space = total_space - used_space

    # Track background
    cv2.rectangle(
        frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + BAR_HEIGHT),
        (45, 45, 50), -1
    )
    cv2.rectangle(
        frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + BAR_HEIGHT),
        (70, 70, 78), 2, cv2.LINE_AA
    )

    # Fill flush against inset on all sides (no gap, no overfill)
    fill_x0 = bar_x + BAR_INSET
    fill_x1 = bar_x + bar_w - BAR_INSET
    fill_y0 = bar_y + BAR_INSET
    fill_y1 = bar_y + BAR_HEIGHT - BAR_INSET
    fill_area_h = fill_y1 - fill_y0

    filled_h = int(fill_area_h * min(used_percent / 100.0, 1.0))
    if used_space > 0 and filled_h < 2:
        filled_h = 2
    used_top = fill_y1 - filled_h  # boundary between used (bottom) and free (top) segments
    # Distinct palette: used = teal/green, free = cool gray (not menu blue)
    used_color_top = (100, 120, 85)
    used_color_bot = (145, 165, 120)
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
        # Thin ridge at boundary (darker teal)
        ridge_color = (60, 85, 70)
        cv2.line(
            frame, (fill_x0, used_top), (fill_x1, used_top),
            ridge_color, 2, cv2.LINE_AA
        )

    # Percent and (size) as a block, centered in each segment
    percent_scale = 0.64
    size_scale = 0.50
    line_gap = 4
    used_pct_str = f"{used_percent:.1f}%" if used_percent >= 0.1 else f"{used_percent:.2f}%"
    free_pct = 100.0 - used_percent
    free_pct_str = f"{free_pct:.1f}%" if free_pct >= 0.1 else f"{free_pct:.2f}%"
    used_size_str = f"({_format_size(used_space)})"
    free_size_str = f"({_format_size(free_space)})"
    (used_tw, used_th), _ = cv2.getTextSize(used_pct_str, font, percent_scale, 2)
    (free_tw, free_th), _ = cv2.getTextSize(free_pct_str, font, percent_scale, 2)
    (used_sw, used_size_th), _ = cv2.getTextSize(used_size_str, font, size_scale, 1)
    (free_sw, free_size_th), _ = cv2.getTextSize(free_size_str, font, size_scale, 1)
    used_cx = (fill_x0 + fill_x1) // 2
    free_cx = (fill_x0 + fill_x1) // 2
    text_color_used = (255, 255, 255)
    text_color_free = (230, 230, 230)
    label_scale = 0.50  # small "FREE" / "USED" at top of each segment
    label_pad = 2
    text_push_up_px = 8  # push % and (size) block up slightly

    if filled_h > 0:
        (used_lw, used_lh), _ = cv2.getTextSize("USED", font, label_scale, 1)
        # USED label at bottom (other end) of used bar
        cv2.putText(
            frame, "USED", (used_cx - used_lw // 2, fill_y1 - label_pad),
            font, label_scale, text_color_used, 1, cv2.LINE_AA
        )
        used_seg_center = (used_top + fill_y1) // 2
        used_block_h = used_th + line_gap + used_size_th
        used_pct_baseline = used_seg_center - used_block_h // 2 + used_th - text_push_up_px
        used_size_baseline = used_pct_baseline + line_gap + used_size_th
        cv2.putText(
            frame, used_pct_str, (used_cx - used_tw // 2, used_pct_baseline),
            font, percent_scale, text_color_used, 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, used_size_str, (used_cx - used_sw // 2, used_size_baseline),
            font, size_scale, text_color_used, 1, cv2.LINE_AA
        )
    free_segment_top = fill_y0
    free_segment_bottom = used_top
    if free_segment_bottom > free_segment_top:
        (free_lw, free_lh), _ = cv2.getTextSize("FREE", font, label_scale, 1)
        cv2.putText(
            frame, "FREE", (free_cx - free_lw // 2, free_segment_top + label_pad + free_lh),
            font, label_scale, text_color_free, 1, cv2.LINE_AA
        )
        free_seg_center = (free_segment_top + free_segment_bottom) // 2
        free_block_h = free_th + line_gap + free_size_th
        free_pct_baseline = free_seg_center - free_block_h // 2 + free_th - text_push_up_px
        free_size_baseline = free_pct_baseline + line_gap + free_size_th
        cv2.putText(
            frame, free_pct_str, (free_cx - free_tw // 2, free_pct_baseline),
            font, percent_scale, text_color_free, 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, free_size_str, (free_cx - free_sw // 2, free_size_baseline),
            font, size_scale, text_color_free, 1, cv2.LINE_AA
        )


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

    _draw_dock_top_rows(frame, dock_x, dock_y, dock_w)

    draw_storage_bar(
        frame, dock_x, dock_w, header_h, items, output_dir
    )

    if button_state.gallery_select_mode:
        if button_state.gallery_priority_modal_open:
            _draw_priority_modal(frame, dock_x, dock_y)
        if button_state.gallery_rename_modal_open:
            _draw_rename_keyboard(frame, dock_x, dock_y)
    else:
        if button_state.gallery_filter_modal_open:
            _draw_filter_modal(frame, dock_x, dock_y)
        if button_state.gallery_sort_modal_open:
            _draw_sort_modal(frame, dock_x, dock_y)
        if button_state.gallery_search_keyboard_open:
            _draw_search_keyboard(frame, dock_x, dock_y)
