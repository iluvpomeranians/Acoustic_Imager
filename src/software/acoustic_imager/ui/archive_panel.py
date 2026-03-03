"""
Archive panel for the gallery grid.

The archive zone permanently occupies the top-right grid slot (row 0,
col cols-1).  It hosts up to MAX_FOLDERS real filesystem subdirectories
under output_dir.  Users drag media thumbnails from the grid and drop
them onto a folder cell to move the file into that directory.

Folder creation: clicking "+" auto-names the next available "Folder N"
slot and creates the subdirectory immediately.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .button import Button, menu_buttons
from ..state import button_state

MAX_FOLDERS = 4

# ── Visual constants ───────────────────────────────────────────────────────────
_ARCHIVE_BG        = (28, 28, 34)
_ARCHIVE_BORDER    = (90, 90, 105)
_ARCHIVE_TITLE_CLR = (200, 200, 210)
_TITLE_H           = 24          # px reserved for "ARCHIVE" title bar
_PAD               = 5           # gap between sub-cells and slot edges

_FOLDER_BG         = (38, 38, 50)
_FOLDER_HOVER_BG   = (55, 75, 115)   # blue tint when a drag is over this cell
_FOLDER_BORDER     = (72, 72, 88)
_FOLDER_HOVER_BDR  = (140, 200, 255)
_FOLDER_NAME_CLR   = (210, 210, 222)
_FOLDER_COUNT_CLR  = (130, 130, 148)
_FOLDER_ICON_CLR   = (175, 135, 55)
_FOLDER_ICON_HOVER = (255, 200, 80)

_ADD_BG            = (32, 36, 46)
_ADD_BORDER        = (70, 100, 155)
_ADD_ICON_CLR      = (110, 155, 215)

_MODAL_BG          = (24, 28, 38)
_MODAL_BORDER      = (200, 200, 200)
_MODAL_TEXT_CLR    = (230, 230, 235)
_MODAL_BTN_CANCEL  = (50, 44, 44)
_MODAL_BTN_CREATE  = (28, 58, 100)


# ── Folder helpers ─────────────────────────────────────────────────────────────

def get_archive_folders(output_dir: Optional[Path]) -> List[Path]:
    """Return up to MAX_FOLDERS subdirectories under *output_dir*, sorted by name."""
    if not output_dir or not output_dir.exists():
        return []
    dirs = sorted(
        (p for p in output_dir.iterdir() if p.is_dir()),
        key=lambda p: p.name.lower(),
    )
    return dirs[:MAX_FOLDERS]


def _next_folder_name(existing: List[Path]) -> str:
    names = {p.name.lower() for p in existing}
    for i in range(1, MAX_FOLDERS + 1):
        candidate = f"Folder {i}"
        if candidate.lower() not in names:
            return candidate
    return "Folder"


def create_archive_folder(output_dir: Optional[Path]) -> Optional[Path]:
    """Auto-name and create the next available folder; return its Path or None."""
    if not output_dir:
        return None
    existing = get_archive_folders(output_dir)
    name = _next_folder_name(existing)
    target = output_dir / name
    try:
        target.mkdir(parents=True, exist_ok=True)
        return target
    except Exception as exc:
        print(f"[archive] create_folder failed: {exc}")
        return None


def move_to_folder(file_path: Path, folder_path: Path) -> bool:
    """Move *file_path* into *folder_path*.  Avoids collisions by appending _N."""
    try:
        dest = folder_path / file_path.name
        if dest.exists() and dest != file_path:
            stem, suffix = file_path.stem, file_path.suffix
            i = 1
            while dest.exists():
                dest = folder_path / f"{stem}_{i}{suffix}"
                i += 1
        shutil.move(str(file_path), str(dest))
        return True
    except Exception as exc:
        print(f"[archive] move_to_folder failed: {exc}")
        return False


# ── Sub-cell geometry ──────────────────────────────────────────────────────────

def slot_cells(x: int, y: int, w: int, h: int) -> List[Tuple[int, int, int, int]]:
    """
    Return the four (cx, cy, cw, ch) sub-cells that fill the archive slot
    (excluding the title bar at the top).
    """
    inner_y = y + _TITLE_H
    inner_h = h - _TITLE_H
    cell_w = (w - 3 * _PAD) // 2
    cell_h = (inner_h - 3 * _PAD) // 2
    cells: List[Tuple[int, int, int, int]] = []
    for row in range(2):
        for col in range(2):
            cx = x + _PAD + col * (cell_w + _PAD)
            cy = inner_y + _PAD + row * (cell_h + _PAD)
            cells.append((cx, cy, cell_w, cell_h))
    return cells


def archive_slot_rect(
    grid_start_x: int,
    grid_start_y: int,
    cols: int,
    thumb_w: int,
    thumb_h: int,
    gap: int,
    scroll_offset: int,
) -> Tuple[int, int, int, int]:
    """Return (x, y, w, h) of the archive slot in screen coordinates."""
    x = grid_start_x + (cols - 1) * (thumb_w + gap)
    y = grid_start_y - scroll_offset
    return x, y, thumb_w, thumb_h


# ── Icon helpers ───────────────────────────────────────────────────────────────

def _draw_folder_icon(
    frame: np.ndarray,
    cx: int,
    cy: int,
    size: int,
    color: Tuple[int, int, int],
) -> None:
    hw = size
    hh = int(size * 0.72)
    tab_w = hw * 6 // 10
    tab_h = max(3, hh // 4)
    tab = np.array([
        [cx - hw, cy - hh // 2 + tab_h],
        [cx - hw, cy - hh // 2],
        [cx - hw + tab_w, cy - hh // 2],
        [cx - hw + tab_w, cy - hh // 2 + tab_h],
    ], dtype=np.int32)
    body = np.array([
        [cx - hw,     cy - hh // 2 + tab_h],
        [cx + hw,     cy - hh // 2 + tab_h],
        [cx + hw,     cy + hh // 2],
        [cx - hw,     cy + hh // 2],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [tab], color)
    cv2.fillPoly(frame, [body], color)
    lighter = tuple(min(255, c + 45) for c in color)
    cv2.polylines(frame, [body], True, lighter, 1, cv2.LINE_AA)  # type: ignore[arg-type]


def _draw_plus_icon(
    frame: np.ndarray,
    cx: int,
    cy: int,
    arm: int,
    thickness: int,
    color: Tuple[int, int, int],
) -> None:
    t = thickness
    cv2.rectangle(frame, (cx - arm, cy - t), (cx + arm, cy + t), color, -1)
    cv2.rectangle(frame, (cx - t, cy - arm), (cx + t, cy + arm), color, -1)


# ── Main draw function ─────────────────────────────────────────────────────────

def draw_archive_slot(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    output_dir: Optional[Path],
    drag_active: bool = False,
    drag_x: int = 0,
    drag_y: int = 0,
) -> None:
    """
    Draw the archive panel in the bounding box (x, y, w, h).

    Registers Button entries in menu_buttons:
      archive_cell_0 .. archive_cell_3  (folder cells)
      archive_add_folder                (the "+" cell)

    Also updates button_state.archive_hover_folder_idx.
    """
    fh, fw = frame.shape[:2]

    # Clip to frame
    if x >= fw or y >= fh or x + w <= 0 or y + h <= 0:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    folders = get_archive_folders(output_dir)
    n = len(folders)

    # ── Slot background ────────────────────────────────────────────────────────
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    frame[y0:y1, x0:x1] = _ARCHIVE_BG
    cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), _ARCHIVE_BORDER, 2, cv2.LINE_AA)

    # ── Title bar ──────────────────────────────────────────────────────────────
    tb_y1 = min(fh, y + _TITLE_H)
    if tb_y1 > y0:
        frame[y0:tb_y1, x0:x1] = (35, 35, 46)
    cv2.line(frame, (x, y + _TITLE_H), (x + w, y + _TITLE_H), _ARCHIVE_BORDER, 1, cv2.LINE_AA)

    title = "ARCHIVE"
    (tw, _), _ = cv2.getTextSize(title, font, 0.40, 1)
    ty = y + _TITLE_H - 7
    if ty > 0:
        cv2.putText(frame, title, (x + (w - tw) // 2, ty),
                    font, 0.40, _ARCHIVE_TITLE_CLR, 1, cv2.LINE_AA)

    # ── Compute hover from drag position ──────────────────────────────────────
    cells = slot_cells(x, y, w, h)
    hover = -1
    if drag_active:
        for i, (cx, cy, cw, ch) in enumerate(cells):
            if i < n and cx <= drag_x < cx + cw and cy <= drag_y < cy + ch:
                hover = i
                break
    button_state.archive_hover_folder_idx = hover

    # ── Draw cells ────────────────────────────────────────────────────────────
    for i, (cx, cy, cw, ch) in enumerate(cells):

        # Clip cell to frame
        if cx >= fw or cy >= fh or cx + cw <= 0 or cy + ch <= 0:
            menu_buttons.pop(f"archive_cell_{i}", None)
            continue

        if i < n:
            # ── Folder cell ───────────────────────────────────────────────────
            folder = folders[i]
            is_hover = (i == hover)
            bg     = _FOLDER_HOVER_BG  if is_hover else _FOLDER_BG
            border = _FOLDER_HOVER_BDR if is_hover else _FOLDER_BORDER
            bw     = 2 if is_hover else 1

            cv2.rectangle(frame, (cx, cy), (cx + cw - 1, cy + ch - 1), bg, -1)
            cv2.rectangle(frame, (cx, cy), (cx + cw - 1, cy + ch - 1), border, bw, cv2.LINE_AA)

            icon_color = _FOLDER_ICON_HOVER if is_hover else _FOLDER_ICON_CLR
            icon_cx = cx + cw // 2
            icon_cy = cy + ch // 2 - 10
            _draw_folder_icon(frame, icon_cx, icon_cy, 11, icon_color)

            # Name (truncated)
            name = folder.name
            if len(name) > 11:
                name = name[:10] + "…"
            (nw, _), _ = cv2.getTextSize(name, font, 0.34, 1)
            if cy + ch - 17 > 0:
                cv2.putText(frame, name, (cx + (cw - nw) // 2, cy + ch - 17),
                            font, 0.34, _FOLDER_NAME_CLR, 1, cv2.LINE_AA)

            # File count
            try:
                nf = sum(1 for p in folder.iterdir() if p.is_file())
            except OSError:
                nf = 0
            cnt = f"{nf} file{'s' if nf != 1 else ''}"
            (cntw, _), _ = cv2.getTextSize(cnt, font, 0.30, 1)
            if cy + ch - 5 > 0:
                cv2.putText(frame, cnt, (cx + (cw - cntw) // 2, cy + ch - 5),
                            font, 0.30, _FOLDER_COUNT_CLR, 1, cv2.LINE_AA)

            key = f"archive_cell_{i}"
            if key not in menu_buttons:
                menu_buttons[key] = Button(cx, cy, cw, ch, folder.name)
            else:
                b = menu_buttons[key]
                b.x, b.y, b.w, b.h, b.text = cx, cy, cw, ch, folder.name

        elif i == n and n < MAX_FOLDERS:
            # ── Add-folder "+" cell ───────────────────────────────────────────
            cv2.rectangle(frame, (cx, cy), (cx + cw - 1, cy + ch - 1), _ADD_BG, -1)
            cv2.rectangle(frame, (cx, cy), (cx + cw - 1, cy + ch - 1), _ADD_BORDER, 1, cv2.LINE_AA)
            _draw_plus_icon(frame, cx + cw // 2, cy + ch // 2, 9, 2, _ADD_ICON_CLR)

            if "archive_add_folder" not in menu_buttons:
                menu_buttons["archive_add_folder"] = Button(cx, cy, cw, ch, "+")
            else:
                b = menu_buttons["archive_add_folder"]
                b.x, b.y, b.w, b.h = cx, cy, cw, ch

        else:
            # Empty / beyond max
            cv2.rectangle(frame, (cx, cy), (cx + cw - 1, cy + ch - 1), (22, 22, 28), -1)
            menu_buttons.pop(f"archive_cell_{i}", None)

    # Remove stale folder buttons if folder count shrank
    for i in range(n, MAX_FOLDERS):
        menu_buttons.pop(f"archive_cell_{i}", None)
    if n >= MAX_FOLDERS:
        menu_buttons.pop("archive_add_folder", None)


# ── Drag ghost ─────────────────────────────────────────────────────────────────

def draw_drag_ghost(
    frame: np.ndarray,
    thumb: Optional[np.ndarray],
    drag_x: int,
    drag_y: int,
    ghost_w: int = 96,
    ghost_h: int = 62,
) -> None:
    """Draw a semi-transparent thumbnail ghost at the drag cursor."""
    if thumb is None:
        return
    try:
        gw, gh = max(10, ghost_w), max(10, ghost_h)
        ghost = cv2.resize(thumb, (gw, gh), interpolation=cv2.INTER_LINEAR)
        gx0, gy0 = drag_x - gw // 2, drag_y - gh // 2
        fh, fw = frame.shape[:2]
        sx0, sy0 = max(0, -gx0), max(0, -gy0)
        dx0, dy0 = max(0, gx0), max(0, gy0)
        dx1, dy1 = min(fw, gx0 + gw), min(fh, gy0 + gh)
        sw, sh = dx1 - dx0, dy1 - dy0
        if sw > 0 and sh > 0:
            roi = frame[dy0:dy1, dx0:dx1]
            g_sl = ghost[sy0:sy0 + sh, sx0:sx0 + sw]
            cv2.addWeighted(g_sl, 0.68, roi, 0.32, 0, dst=roi)
            cv2.rectangle(frame, (dx0, dy0), (dx1 - 1, dy1 - 1),
                          (140, 200, 255), 1, cv2.LINE_AA)
    except Exception:
        pass
