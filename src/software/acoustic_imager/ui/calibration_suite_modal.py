"""
Calibration Suite modal: run calibration tests (config, SPI, mic mapping) and show live log.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import cv2
import numpy as np
from pathlib import Path

from . import ui_cache
from .button import menu_buttons, Button
from ..state import button_state
from ..config import MENU_ACTIVE_BLUE

MODAL_W = 560
MODAL_H = 420
PAD = 20
HEADER_H = 50
LOG_LINE_H = 18
LOG_MAX_LINES = 14  # visible lines in log area
START_BTN_H = 44
SCROLLBAR_W = 12

# Last-drawn scrollbar thumb (log_top, thumb_y, thumb_h, log_h, max_scroll) for click-to-page
_cal_suite_scroll_thumb_geom: tuple = (0, 0, 0, 0, 0)

# Log content area (text only, not scrollbar) for touch-drag scroll
LOG_AREA_TOP = HEADER_H + START_BTN_H + PAD
LOG_AREA_H = MODAL_H - LOG_AREA_TOP - 50
LOG_AREA_W = MODAL_W - 2 * PAD - SCROLLBAR_W


def _log_content_bounds(fw: int, fh: int) -> tuple[int, int, int, int]:
    """Return (x, y, w, h) of log content area (text region, not scrollbar) in screen coords."""
    modal_x = (fw - MODAL_W) // 2
    modal_y = (fh - MODAL_H) // 2
    x = modal_x + PAD
    y = modal_y + LOG_AREA_TOP
    return (x, y, LOG_AREA_W, LOG_AREA_H)


def _is_in_log_content_area(x: int, y: int, fw: int, fh: int) -> bool:
    """True if (x,y) is in the log text area (not scrollbar, not Start/Close)."""
    lx, ly, lw, lh = _log_content_bounds(fw, fh)
    return lx <= x < lx + lw and ly <= y < ly + lh


def _hit_any_cal_suite_button(x: int, y: int) -> bool:
    """True if (x,y) hits Start, Stop, Close, or scrollbar buttons/track."""
    for k in ("cal_suite_start", "cal_suite_stop", "cal_suite_close", "cal_suite_scroll_up", "cal_suite_scroll_down", "cal_suite_scroll_track"):
        if k in menu_buttons:
            b = menu_buttons[k]
            if b.w > 0 and b.h > 0 and b.contains(x, y):
                return True
    return False


def _repo_root() -> Path:
    """Return repo root (parent of src). Modal is at .../src/software/acoustic_imager/ui/."""
    return Path(__file__).resolve().parents[4]


def _append_log(line: str) -> None:
    button_state.calibration_suite_log.append(line)


def _run_calibration_suite() -> None:
    """Run full calibration suite (0-6) in background; append lines to button_state.calibration_suite_log.
    Uses Popen so the main thread can terminate the process when user clicks Stop calibrating."""
    proc = None
    try:
        root = _repo_root()
        run_suite_py = root / "utilities" / "calibration" / "run_suite.py"
        if not run_suite_py.exists():
            _append_log("run_suite.py not found at %s" % run_suite_py)
            return
        gain = (button_state.gain_mode or "HIGH").lower()
        if gain not in ("high", "low"):
            gain = "high"
        cmd = [sys.executable, str(run_suite_py), "--gain", gain]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(root),
        )
        button_state.calibration_suite_process = proc
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                _append_log(line[:90].rstrip())
        try:
            err = proc.stderr.read()
        except Exception:
            err = ""
        for line in (err or "").splitlines():
            _append_log("[stderr] " + line[:85].rstrip())
        if proc.returncode != 0 and proc.returncode is not None:
            _append_log("Suite exited with code %d" % proc.returncode)
    except Exception as e:
        _append_log("Suite error: %s" % e)
    finally:
        button_state.calibration_suite_process = None
        button_state.calibration_suite_running = False
        # Write log to fixed dump file (matches what the user saw in the modal)
        dump_path = _repo_root() / "utilities" / "calibration" / "calibration_suite_cal_dump.txt"
        try:
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            dump_path.write_text("\n".join(button_state.calibration_suite_log), encoding="utf-8")
        except Exception:
            pass


def _start_suite() -> None:
    if button_state.calibration_suite_running:
        return
    button_state.calibration_suite_log.clear()
    button_state.calibration_suite_running = True
    button_state.calibration_suite_scroll_offset = 0
    t = threading.Thread(target=_run_calibration_suite, daemon=True)
    t.start()


def draw_calibration_suite_modal(frame: np.ndarray) -> None:
    """Draw Calibration Suite modal: title, Start button, scrollable log, Close."""
    if not button_state.calibration_suite_modal_open:
        return

    fh, fw = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)
    border_color = (100, 100, 100)
    dim_color = (180, 180, 180)

    # Lighter dim so the cached last frame remains visible behind the modal
    ui_cache.apply_modal_dim(frame, 0.15)

    modal_x = (fw - MODAL_W) // 2
    modal_y = (fh - MODAL_H) // 2

    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + MODAL_W, modal_y + MODAL_H), (40, 40, 40), -1)
    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + MODAL_W, modal_y + MODAL_H), border_color, 3, cv2.LINE_AA)

    # Title
    cv2.putText(frame, "Calibration Suite", (modal_x + PAD, modal_y + 34), font, 0.68, text_color, 1, cv2.LINE_AA)

    # Start Calibration button
    btn_y = modal_y + HEADER_H
    row_w = MODAL_W - 2 * PAD - SCROLLBAR_W
    running = button_state.calibration_suite_running
    if running:
        cv2.rectangle(frame, (modal_x + PAD, btn_y), (modal_x + PAD + row_w, btn_y + START_BTN_H), (60, 60, 60), -1, cv2.LINE_AA)
        cv2.putText(frame, "Running...", (modal_x + (MODAL_W - 70) // 2, btn_y + START_BTN_H // 2 + 6), font, 0.50, dim_color, 1, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (modal_x + PAD, btn_y), (modal_x + PAD + row_w, btn_y + START_BTN_H), MENU_ACTIVE_BLUE, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (modal_x + PAD, btn_y), (modal_x + PAD + row_w, btn_y + START_BTN_H), (120, 160, 255), 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize("Start Calibration", font, 0.54, 1)
        cv2.putText(frame, "Start Calibration", (modal_x + (MODAL_W - tw) // 2, btn_y + START_BTN_H // 2 + 6), font, 0.54, text_color, 1, cv2.LINE_AA)

    if "cal_suite_start" not in menu_buttons:
        menu_buttons["cal_suite_start"] = Button(0, 0, row_w, START_BTN_H, "")
    menu_buttons["cal_suite_start"].x = modal_x + PAD
    menu_buttons["cal_suite_start"].y = btn_y
    menu_buttons["cal_suite_start"].w = row_w
    menu_buttons["cal_suite_start"].h = START_BTN_H

    # Log area (scrollable)
    log_top = btn_y + START_BTN_H + PAD
    log_h = MODAL_H - (log_top - modal_y) - 50  # leave room for close
    log_w = MODAL_W - 2 * PAD - SCROLLBAR_W
    cv2.rectangle(frame, (modal_x + PAD, log_top), (modal_x + PAD + log_w, log_top + log_h), (25, 25, 25), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (modal_x + PAD, log_top), (modal_x + PAD + log_w, log_top + log_h), (70, 70, 70), 1, cv2.LINE_AA)

    log_lines = button_state.calibration_suite_log
    total_log_h = len(log_lines) * LOG_LINE_H
    max_scroll = max(0, total_log_h - log_h)
    scroll = min(button_state.calibration_suite_scroll_offset, max_scroll)
    button_state.calibration_suite_scroll_offset = scroll

    # Draw visible log lines
    start_idx = scroll // LOG_LINE_H
    y_off = log_top + 6 - (scroll % LOG_LINE_H)
    for i in range(start_idx, len(log_lines)):
        ly = y_off + (i - start_idx) * LOG_LINE_H
        if ly < log_top or ly + LOG_LINE_H > log_top + log_h:
            continue
        line = log_lines[i]
        if len(line) > 52:
            line = line[:49] + "..."
        cv2.putText(frame, line, (modal_x + PAD + 4, ly + 14), font, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

    # Scrollbar for log
    sb_x = modal_x + MODAL_W - SCROLLBAR_W - PAD
    if max_scroll > 0:
        cv2.rectangle(frame, (sb_x, log_top), (sb_x + SCROLLBAR_W, log_top + log_h), (50, 50, 50), -1, cv2.LINE_AA)
        thumb_ratio = log_h / total_log_h
        thumb_h = max(24, int(log_h * thumb_ratio))
        thumb_y = log_top + int((log_h - thumb_h) * scroll / max_scroll) if max_scroll > 0 else log_top
        cv2.rectangle(frame, (sb_x + 2, thumb_y), (sb_x + SCROLLBAR_W - 2, thumb_y + thumb_h), (110, 110, 110), -1, cv2.LINE_AA)
        # Up/down arrow buttons (restore w/h every frame so they stay clickable after being zeroed)
        if "cal_suite_scroll_up" not in menu_buttons:
            menu_buttons["cal_suite_scroll_up"] = Button(sb_x, log_top, SCROLLBAR_W, 24, "")
            menu_buttons["cal_suite_scroll_down"] = Button(sb_x, log_top + log_h - 24, SCROLLBAR_W, 24, "")
        menu_buttons["cal_suite_scroll_up"].x, menu_buttons["cal_suite_scroll_up"].y = sb_x, log_top
        menu_buttons["cal_suite_scroll_up"].w, menu_buttons["cal_suite_scroll_up"].h = SCROLLBAR_W, 24
        menu_buttons["cal_suite_scroll_down"].x, menu_buttons["cal_suite_scroll_down"].y = sb_x, log_top + log_h - 24
        menu_buttons["cal_suite_scroll_down"].w, menu_buttons["cal_suite_scroll_down"].h = SCROLLBAR_W, 24
        # Full track for click-to-page and thumb drag (store thumb_y/h for click handler)
        if "cal_suite_scroll_track" not in menu_buttons:
            menu_buttons["cal_suite_scroll_track"] = Button(sb_x, log_top, SCROLLBAR_W, log_h, "")
        menu_buttons["cal_suite_scroll_track"].x, menu_buttons["cal_suite_scroll_track"].y = sb_x, log_top
        menu_buttons["cal_suite_scroll_track"].w, menu_buttons["cal_suite_scroll_track"].h = SCROLLBAR_W, log_h
        # Store for click handler: (log_top, thumb_y, thumb_h, log_h, max_scroll)
        global _cal_suite_scroll_thumb_geom
        _cal_suite_scroll_thumb_geom = (log_top, thumb_y, thumb_h, log_h, max_scroll)
    else:
        if "cal_suite_scroll_up" in menu_buttons:
            menu_buttons["cal_suite_scroll_up"].w = menu_buttons["cal_suite_scroll_down"].w = 0
            menu_buttons["cal_suite_scroll_up"].h = menu_buttons["cal_suite_scroll_down"].h = 0
        if "cal_suite_scroll_track" in menu_buttons:
            menu_buttons["cal_suite_scroll_track"].w = menu_buttons["cal_suite_scroll_track"].h = 0

    # Stop calibrating (red, bottom left) – only when running
    stop_btn_w, stop_btn_h = 140, 36
    stop_btn_x = modal_x + PAD
    stop_btn_y = modal_y + MODAL_H - stop_btn_h - PAD
    STOP_RED = (0, 0, 255)
    STOP_RED_LIGHT = (100, 100, 255)
    if running:
        cv2.rectangle(frame, (stop_btn_x, stop_btn_y), (stop_btn_x + stop_btn_w, stop_btn_y + stop_btn_h), STOP_RED, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (stop_btn_x, stop_btn_y), (stop_btn_x + stop_btn_w, stop_btn_y + stop_btn_h), STOP_RED_LIGHT, 1, cv2.LINE_AA)
        (tw_stop, _), _ = cv2.getTextSize("Stop calibrating", font, 0.48, 1)
        cv2.putText(frame, "Stop calibrating", (stop_btn_x + (stop_btn_w - tw_stop) // 2, stop_btn_y + stop_btn_h // 2 + 6), font, 0.48, text_color, 1, cv2.LINE_AA)
    if "cal_suite_stop" not in menu_buttons:
        menu_buttons["cal_suite_stop"] = Button(stop_btn_x, stop_btn_y, stop_btn_w, stop_btn_h, "")
    menu_buttons["cal_suite_stop"].x, menu_buttons["cal_suite_stop"].y = stop_btn_x, stop_btn_y
    menu_buttons["cal_suite_stop"].w = stop_btn_w if running else 0
    menu_buttons["cal_suite_stop"].h = stop_btn_h if running else 0

    # Close button (bottom right)
    close_w, close_h = 72, 28
    close_x = modal_x + MODAL_W - close_w - PAD
    close_y = modal_y + MODAL_H - close_h - PAD
    if "cal_suite_close" not in menu_buttons:
        menu_buttons["cal_suite_close"] = Button(close_x, close_y, close_w, close_h, "Close")
    else:
        menu_buttons["cal_suite_close"].x, menu_buttons["cal_suite_close"].y = close_x, close_y
        menu_buttons["cal_suite_close"].w, menu_buttons["cal_suite_close"].h = close_w, close_h
    menu_buttons["cal_suite_close"].draw(frame, transparent=True, active_color=MENU_ACTIVE_BLUE)

    if "cal_suite_modal_panel" not in menu_buttons:
        menu_buttons["cal_suite_modal_panel"] = Button(modal_x, modal_y, MODAL_W, MODAL_H, "")
    else:
        b = menu_buttons["cal_suite_modal_panel"]
        b.x, b.y, b.w, b.h = modal_x, modal_y, MODAL_W, MODAL_H


def handle_calibration_suite_modal_click(x: int, y: int) -> bool:
    """Handle click. Returns True if handled."""
    if not button_state.calibration_suite_modal_open:
        return False

    if "cal_suite_close" in menu_buttons and menu_buttons["cal_suite_close"].contains(x, y):
        button_state.calibration_suite_modal_open = False
        return True

    # Stop calibrating (terminate subprocess)
    if button_state.calibration_suite_running and "cal_suite_stop" in menu_buttons and menu_buttons["cal_suite_stop"].w > 0 and menu_buttons["cal_suite_stop"].contains(x, y):
        proc = getattr(button_state, "calibration_suite_process", None)
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
        _append_log("Stopped by user")
        return True

    if not button_state.calibration_suite_running and "cal_suite_start" in menu_buttons and menu_buttons["cal_suite_start"].contains(x, y):
        _start_suite()
        return True

    # Scroll: up button, down button, then track (click above/below thumb = page)
    log_area_top = HEADER_H + START_BTN_H + PAD
    log_area_h = MODAL_H - log_area_top - 50
    total_log_h = len(button_state.calibration_suite_log) * LOG_LINE_H
    max_scroll = max(0, total_log_h - log_area_h)
    page_step = max(1, log_area_h // 2)  # about half a screen

    if "cal_suite_scroll_up" in menu_buttons and menu_buttons["cal_suite_scroll_up"].w > 0 and menu_buttons["cal_suite_scroll_up"].contains(x, y):
        button_state.calibration_suite_scroll_offset = max(0, button_state.calibration_suite_scroll_offset - page_step)
        return True
    if "cal_suite_scroll_down" in menu_buttons and menu_buttons["cal_suite_scroll_down"].w > 0 and menu_buttons["cal_suite_scroll_down"].contains(x, y):
        button_state.calibration_suite_scroll_offset = min(max_scroll, button_state.calibration_suite_scroll_offset + page_step)
        return True
    # Click on scrollbar track (above or below thumb): page up/down
    if max_scroll > 0 and "cal_suite_scroll_track" in menu_buttons and menu_buttons["cal_suite_scroll_track"].w > 0 and menu_buttons["cal_suite_scroll_track"].contains(x, y):
        log_top, thumb_y, thumb_h, log_h, _ = _cal_suite_scroll_thumb_geom
        if y < thumb_y:
            button_state.calibration_suite_scroll_offset = max(0, button_state.calibration_suite_scroll_offset - page_step)
        elif y > thumb_y + thumb_h:
            button_state.calibration_suite_scroll_offset = min(max_scroll, button_state.calibration_suite_scroll_offset + page_step)
        return True

    if "cal_suite_modal_panel" in menu_buttons and menu_buttons["cal_suite_modal_panel"].contains(x, y):
        return True

    button_state.calibration_suite_modal_open = False
    return True


def handle_calibration_suite_modal_scroll(delta: int) -> bool:
    """Handle mouse wheel. delta > 0 = scroll down. Returns True if handled."""
    if not button_state.calibration_suite_modal_open:
        return False
    log_area_top = HEADER_H + START_BTN_H + PAD
    log_area_h = MODAL_H - log_area_top - 50
    total_log_h = len(button_state.calibration_suite_log) * LOG_LINE_H
    max_scroll = max(0, total_log_h - log_area_h)
    if max_scroll <= 0:
        return False
    step = 30
    new_offset = button_state.calibration_suite_scroll_offset + delta
    button_state.calibration_suite_scroll_offset = max(0, min(max_scroll, new_offset))
    return True


def handle_calibration_suite_modal_mouse(event: int, x: int, y: int, fw: int, fh: int) -> bool:
    """Handle mouse/touch events for log area drag-to-scroll. Returns True if consumed."""
    if not button_state.calibration_suite_modal_open:
        return False

    total_log_h = len(button_state.calibration_suite_log) * LOG_LINE_H
    max_scroll = max(0, total_log_h - LOG_AREA_H)

    if event == cv2.EVENT_LBUTTONDOWN:
        if _is_in_log_content_area(x, y, fw, fh) and not _hit_any_cal_suite_button(x, y):
            button_state.calibration_suite_log_dragging = True
            button_state.calibration_suite_log_drag_start_y = y
            button_state.calibration_suite_log_drag_start_scroll = button_state.calibration_suite_scroll_offset
            return True
        return False

    if event == cv2.EVENT_MOUSEMOVE:
        if button_state.calibration_suite_log_dragging:
            dy = y - button_state.calibration_suite_log_drag_start_y
            if abs(dy) > ui_cache.DRAG_PX:
                new_scroll = button_state.calibration_suite_log_drag_start_scroll - dy
                button_state.calibration_suite_scroll_offset = max(0, min(int(new_scroll), max_scroll))
            return True
        return False

    if event == cv2.EVENT_LBUTTONUP:
        if button_state.calibration_suite_log_dragging:
            button_state.calibration_suite_log_dragging = False
            return True
        return False

    return False
