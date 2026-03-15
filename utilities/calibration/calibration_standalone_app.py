#!/usr/bin/env python3
"""
Standalone calibration suite app: single window, runs full suite (main app must be stopped).
Shows silence countdown during step 0, then log and PASSED/FAILED. At end, restarts the main app.

Usage (from repo root; usually launched by main app):
  python3 utilities/calibration/calibration_standalone_app.py [--restart-main] [--gain high|low]
When --restart-main is set, restarts the acoustic imager main app when user dismisses the result.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

_CAL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CAL_DIR.parents[1]
_MAIN_PY = _REPO_ROOT / "src" / "software" / "acoustic_imager" / "main.py"

# Fullscreen (no desktop visible) — title and log use entire screen
FS_W = 1920
FS_H = 1080
PAD = 28
HEADER_H = 72
LOG_LINE_H = 34
LOG_TOP = HEADER_H + PAD
LOG_H = FS_H - LOG_TOP - 40
LOG_W = FS_W - 2 * PAD - 20
SCROLLBAR_W = 24

STEP_LABELS = {
    "silence": "Silence check",
    "pins": "Pin monitor",
    "geometry": "Geometry",
    "spi_devices": "SPI devices",
    "parsing_ok": "Parsing & rates",
    "metrics": "Metrics",
    "post_gain": "Post-gain",
}


def _read_failed_steps() -> list[str]:
    path = _CAL_DIR / "calibration_suite_last_result.txt"
    failed = []
    try:
        if not path.exists():
            return failed
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            if key in STEP_LABELS and val.strip().lower() in ("false", "0"):
                failed.append(STEP_LABELS[key])
    except Exception:
        pass
    return failed


def _run_suite(gain: str, log_lines: list[str], silence_start_callback: callable, silence_triggered: list) -> None:
    run_suite_py = _CAL_DIR / "run_suite.py"
    cmd = [sys.executable, str(run_suite_py), "--gain", gain]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=str(_REPO_ROOT),
    )
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            line = line.rstrip()
            log_lines.append(line[:400])  # keep enough for full display and dump
            m = re.match(r"SILENCE_START\s+([\d.]+)", line.strip())
            if m:
                silence_start_callback(float(m.group(1)))
                silence_triggered[0] = True
            elif "[0] Silence check" in line and not silence_triggered[0]:
                silence_start_callback(2.0)  # fallback if SILENCE_START was buffered
                silence_triggered[0] = True
    try:
        err = proc.stderr.read()
    except Exception:
        err = ""
    for line in (err or "").splitlines():
        log_lines.append("[stderr] " + line[:380].rstrip())
    if proc.returncode is not None and proc.returncode != 0:
        log_lines.append("Suite exited with code %d" % proc.returncode)
    # Always dump log when suite finishes (same path as main-thread write)
    try:
        (_CAL_DIR / "calibration_suite_cal_dump.txt").write_text("\n".join(log_lines), encoding="utf-8")
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Standalone calibration suite (stops main app; restarts when done)")
    ap.add_argument("--restart-main", action="store_true", help="Restart acoustic imager main app when done")
    ap.add_argument("--gain", choices=("high", "low"), default="high", help="Gain for metrics step")
    args = ap.parse_args()

    log_lines: list[str] = []
    scroll_offset = 0
    max_scroll = 0
    suite_done = False
    result_passed = False
    result_count = 0
    result_total = 0
    show_result_overlay = False
    ok_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
    silence_remaining_sec: float | None = None  # set by thread when SILENCE_START seen; main loop counts down
    silence_triggered: list[bool] = [False]  # so we only start countdown once (SILENCE_START or "[0] Silence check")
    user_dismissed_result: list[bool] = [False]  # list so closure can set from mouse callback

    def on_silence_start(sec: float) -> None:
        nonlocal silence_remaining_sec
        silence_remaining_sec = sec

    def worker() -> None:
        nonlocal suite_done
        _run_suite(args.gain, log_lines, on_silence_start, silence_triggered)
        suite_done = True

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    cv2.namedWindow("Calibration Suite", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration Suite", FS_W, FS_H)
    cv2.setWindowProperty("Calibration Suite", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def _on_mouse(event: int, x: int, y: int, _a: int, _b: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and show_result_overlay and ok_rect[2] > 0:
            ox, oy, ow, oh = ok_rect
            if ox <= x < ox + ow and oy <= y < oy + oh:
                user_dismissed_result[0] = True

    cv2.setMouseCallback("Calibration Suite", _on_mouse)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)
    blue = (255, 180, 80)
    green = (100, 255, 0)
    red = (0, 100, 255)

    LOG_FONT_SCALE = 0.72
    LOG_FONT_THICKNESS = 2
    # Compute max chars that fit in LOG_W so lines are not cut off before the right edge
    (char_w, _), _ = cv2.getTextSize("W", font, LOG_FONT_SCALE, LOG_FONT_THICKNESS)
    LOG_MAX_CHARS = max(60, (LOG_W - 24) // char_w) if char_w > 0 else 120

    while True:
        frame = np.zeros((FS_H, FS_W, 3), dtype=np.uint8)
        frame[:] = (28, 28, 28)
        cv2.rectangle(frame, (0, 0), (FS_W, FS_H), (70, 70, 70), 2, cv2.LINE_AA)

        cv2.putText(frame, "Calibration Suite", (PAD, 52), font, 1.25, text_color, 2, cv2.LINE_AA)

        total_log_h = len(log_lines) * LOG_LINE_H
        max_scroll = max(0, total_log_h - LOG_H)
        scroll_offset = min(scroll_offset, max_scroll)
        start_idx = scroll_offset // LOG_LINE_H
        y_off = LOG_TOP + 8 - (scroll_offset % LOG_LINE_H)
        cv2.rectangle(frame, (PAD, LOG_TOP), (PAD + LOG_W, LOG_TOP + LOG_H), (18, 18, 18), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (PAD, LOG_TOP), (PAD + LOG_W, LOG_TOP + LOG_H), (60, 60, 60), 1, cv2.LINE_AA)
        for i in range(start_idx, len(log_lines)):
            ly = y_off + (i - start_idx) * LOG_LINE_H
            if ly < LOG_TOP or ly + LOG_LINE_H > LOG_TOP + LOG_H:
                continue
            line = log_lines[i]
            if len(line) > LOG_MAX_CHARS:
                line = line[: LOG_MAX_CHARS - 3] + "..."
            cv2.putText(frame, line, (PAD + 10, ly + LOG_LINE_H - 8), font, LOG_FONT_SCALE, (245, 245, 245), LOG_FONT_THICKNESS, cv2.LINE_AA)

        if total_log_h > LOG_H:
            sb_x = FS_W - PAD - SCROLLBAR_W
            cv2.rectangle(frame, (sb_x, LOG_TOP), (sb_x + SCROLLBAR_W, LOG_TOP + LOG_H), (45, 45, 45), -1, cv2.LINE_AA)
            thumb_ratio = LOG_H / total_log_h
            thumb_h = max(40, int(LOG_H * thumb_ratio))
            thumb_y = LOG_TOP + int((LOG_H - thumb_h) * scroll_offset / max_scroll) if max_scroll > 0 else LOG_TOP
            cv2.rectangle(frame, (sb_x + 3, thumb_y), (sb_x + SCROLLBAR_W - 3, thumb_y + thumb_h), (120, 120, 120), -1, cv2.LINE_AA)

        # Silence countdown overlay (centered on full screen, large and visible)
        if silence_remaining_sec is not None and silence_remaining_sec > 0:
            sec_int = int(math.ceil(silence_remaining_sec))
            msg = "Stay quiet: %d sec" % sec_int
            (tw, th), _ = cv2.getTextSize(msg, font, 1.2, 2)
            cx, cy = FS_W // 2, FS_H // 2
            bw, bh = tw + 80, th + 50
            bx, by = cx - bw // 2, cy - bh // 2
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (30, 60, 30), -1, cv2.LINE_AA)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 200, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, msg, (cx - tw // 2, cy + th // 2 - 5), font, 1.2, (0, 255, 100), 2, cv2.LINE_AA)

        # When suite just finished, parse and show result
        if suite_done and not show_result_overlay:
            done_match = None
            for line in reversed(log_lines):
                m = re.search(r"=== Done: (\d+)/(\d+) steps passed ===", line)
                if m:
                    done_match = m
                    break
            if done_match:
                result_count = int(done_match.group(1))
                result_total = int(done_match.group(2))
                result_passed = result_count == result_total
            show_result_overlay = True
            dump_path = _CAL_DIR / "calibration_suite_cal_dump.txt"
            try:
                dump_path.write_text("\n".join(log_lines), encoding="utf-8")
            except Exception:
                pass

        # Result overlay (centered on fullscreen, larger text)
        if show_result_overlay:
            res_w, res_h = 420, 200
            res_x = (FS_W - res_w) // 2
            res_y = (FS_H - res_h) // 2
            cv2.rectangle(frame, (res_x, res_y), (res_x + res_w, res_y + res_h), (35, 35, 35), -1, cv2.LINE_AA)
            cv2.rectangle(frame, (res_x, res_y), (res_x + res_w, res_y + res_h), (140, 140, 140), 2, cv2.LINE_AA)
            if result_passed:
                msg = "PASSED (%d/%d steps)" % (result_count, result_total)
                msg_color = green
            else:
                msg = "FAILED (%d/%d steps)" % (result_count, result_total)
                msg_color = red
            (tw, _), _ = cv2.getTextSize(msg, font, 1.0, 2)
            cv2.putText(frame, msg, (res_x + (res_w - tw) // 2, res_y + 52), font, 1.0, msg_color, 2, cv2.LINE_AA)
            failed_steps = _read_failed_steps()
            if failed_steps:
                failed_str = ", ".join(failed_steps)[:45]
                cv2.putText(frame, "Failed: " + failed_str, (res_x + 16, res_y + 92), font, 0.52, (220, 160, 160), 2, cv2.LINE_AA)
            ok_w, ok_h = 120, 48
            ok_x = res_x + (res_w - ok_w) // 2
            ok_y = res_y + res_h - ok_h - 24
            ok_rect = (ok_x, ok_y, ok_w, ok_h)
            cv2.rectangle(frame, (ok_x, ok_y), (ok_x + ok_w, ok_y + ok_h), blue, -1, cv2.LINE_AA)
            cv2.rectangle(frame, (ok_x, ok_y), (ok_x + ok_w, ok_y + ok_h), (200, 220, 255), 2, cv2.LINE_AA)
            (tw_ok, _), _ = cv2.getTextSize("OK", font, 0.8, 2)
            cv2.putText(frame, "OK", (ok_x + (ok_w - tw_ok) // 2, ok_y + ok_h // 2 + 8), font, 0.8, text_color, 2, cv2.LINE_AA)

        cv2.imshow("Calibration Suite", frame)
        key = cv2.waitKey(50) & 0xFF
        if key == 27:
            break
        if user_dismissed_result[0]:
            break
        if show_result_overlay and key in (ord("\r"), ord("\n"), ord(" "), ord("o"), ord("O")):
            break

        # Count down silence timer (main thread drives time)
        if silence_remaining_sec is not None:
            silence_remaining_sec -= 0.05
            if silence_remaining_sec <= 0:
                silence_remaining_sec = None

    cv2.destroyAllWindows()

    if args.restart_main and _MAIN_PY.exists():
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_REPO_ROOT / "src" / "software")
        try:
            subprocess.Popen(
                [sys.executable, str(_MAIN_PY)],
                cwd=str(_REPO_ROOT),
                env=env,
                start_new_session=True,
            )
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
