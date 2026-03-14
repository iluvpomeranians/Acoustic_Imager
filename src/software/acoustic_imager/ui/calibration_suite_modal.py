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


def _repo_root() -> Path:
    """Return repo root (parent of src). Modal is at .../src/software/acoustic_imager/ui/."""
    return Path(__file__).resolve().parents[4]


def _append_log(line: str) -> None:
    button_state.calibration_suite_log.append(line)


def _run_calibration_suite() -> None:
    """Run calibration checks in background; append lines to button_state.calibration_suite_log."""
    try:
        root = _repo_root()
        _append_log("=== Calibration Suite ===")

        # 1) Config / calibration_test validation
        _append_log("[1/4] Config & calibration test setup...")
        cal_script = root / "utilities" / "calibration" / "calibration_test.py"
        if cal_script.exists():
            r = subprocess.run(
                [sys.executable, str(cal_script), "--validate"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(root),
            )
            for line in (r.stdout or "").strip().splitlines():
                _append_log("  " + line)
            for line in (r.stderr or "").strip().splitlines():
                _append_log("  [stderr] " + line)
            if r.returncode != 0:
                _append_log("  FAILED (exit %d)" % r.returncode)
            else:
                _append_log("  OK")
        else:
            _append_log("  calibration_test.py not found")

        # 2) Array geometry / mic mapping (in-process)
        _append_log("[2/4] Mic array geometry (config)...")
        try:
            from acoustic_imager import config
            n = getattr(config, "N_MICS", 16)
            x = getattr(config, "x_coords_hw", None)
            y = getattr(config, "y_coords_hw", None)
            if x is None or y is None:
                _append_log("  Missing x_coords_hw or y_coords_hw")
            elif len(x) != n or len(y) != n:
                _append_log("  Length mismatch: expected %d, got %d / %d" % (n, len(x), len(y)))
            else:
                _append_log("  OK (%d mics)" % n)
        except Exception as e:
            _append_log("  Error: %s" % e)

        # 3) SPI device nodes
        _append_log("[3/4] SPI devices...")
        spi_script = root / "utilities" / "calibration" / "check_spi_devices.sh"
        if spi_script.exists():
            r = subprocess.run(
                ["bash", str(spi_script)],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=str(root),
            )
            for line in (r.stdout or "").strip().splitlines():
                _append_log("  " + line[:80])
            if r.returncode != 0 and (r.stderr or "").strip():
                for line in (r.stderr or "").strip().splitlines():
                    _append_log("  " + line[:80])
        else:
            # Fallback: check /dev/spidev* exists
            import glob
            devs = glob.glob("/dev/spidev*")
            if devs:
                _append_log("  Found: %s" % " ".join(devs))
            else:
                _append_log("  No /dev/spidev* found")

        # 4) SPI data (magic probe) - optional; may fail if SPI in use by app
        _append_log("[4/4] SPI data (magic probe, may skip if bus in use)...")
        probe_script = root / "utilities" / "calibration" / "spi_magic_probe.py"
        if probe_script.exists():
            r = subprocess.run(
                [sys.executable, str(probe_script), "--no-sync"],
                capture_output=True,
                text=True,
                timeout=8,
                cwd=str(root),
            )
            for line in (r.stdout or "").strip().splitlines():
                _append_log("  " + line[:80])
            if r.returncode != 0 and (r.stderr or "").strip():
                _append_log("  (probe returned %d)" % r.returncode)
        else:
            _append_log("  spi_magic_probe.py not found")

        _append_log("=== Done ===")
    except Exception as e:
        _append_log("Suite error: %s" % e)
    finally:
        button_state.calibration_suite_running = False


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

    ui_cache.apply_modal_dim(frame, 0.5)

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

    # Close button
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
