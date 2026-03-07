"""
Spectrum analyzer overlay: frequency bar with smooth spectrum curve.

Two modes (MENU: SPECTRUM: dB / SPECTRUM: NORM):
- dB: amplitude in dB (relative to peak), with bottom ruler.
- NORM: normalized amplitude (peak = 100%, power^0.4), no ruler.

Uses the same FFT data as the beamforming pipeline. Bandpass overlay always drawn.
"""

from __future__ import annotations

import numpy as np
import cv2

from ..config import (
    FREQ_BAR_BLUE,
    REL_DB_MIN,
    REL_DB_MAX,
)
from .bars import _get_panel_bg, freq_to_y


SPECTRUM_DB_MIN = REL_DB_MIN
SPECTRUM_DB_MAX = REL_DB_MAX

BAR_MARGIN_LEFT = 8
BAR_MARGIN_RIGHT = 5
RULER_HEIGHT = 20   # bottom strip for dB ruler (dB mode only)

GRAPH_COLOR_TOP = (0, 220, 255)
GRAPH_COLOR_BOT = (0, 120, 255)
GRAPH_DIM_TOP = (140, 160, 180)
GRAPH_DIM_BOT = (90, 120, 150)
SPECTRUM_CURVE_BGR = (255, 255, 200)
SPECTRUM_CURVE_THICKNESS = 2
DB_LABEL_COLOR = (200, 200, 200)
RULER_TICK_COLOR = (180, 180, 180)


def _draw_db_ruler_bottom(bar: np.ndarray, bar_w: int, graph_h: int) -> None:
    """Draw a horizontal dB ruler at the bottom; labels are rotated 90° CCW so they read vertically."""
    h = bar.shape[0]
    graph_left = BAR_MARGIN_LEFT
    graph_right = bar_w - BAR_MARGIN_RIGHT
    graph_width = max(1, graph_right - graph_left)
    db_span = SPECTRUM_DB_MAX - SPECTRUM_DB_MIN

    # Ruler baseline at the very bottom edge of the bar
    y_baseline = h - 1
    tick_len_px = 8   # short tick stubs above the baseline
    y_tick_top = y_baseline - tick_len_px
    y_tick_bottom = y_baseline
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    thickness = 1

    cv2.line(bar, (graph_left, y_baseline), (graph_right, y_baseline), RULER_TICK_COLOR, 1, cv2.LINE_AA)

    for db in range(int(SPECTRUM_DB_MIN), int(SPECTRUM_DB_MAX) + 1, 10):
        frac = (db - SPECTRUM_DB_MIN) / db_span
        x = int(graph_left + frac * graph_width)
        x = int(np.clip(x, graph_left, graph_right))
        cv2.line(bar, (x, y_tick_top), (x, y_tick_bottom), RULER_TICK_COLOR, 1, cv2.LINE_AA)
        label = f"{db}" if db != 0 else "0"
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        pad = 2
        # Draw label on small canvas, rotate 90° CCW, then paste so it reads vertically
        canvas_w = tw + 2 * pad
        canvas_h = th + 2 * pad
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        cv2.putText(canvas, label, (pad, pad + th), font, font_scale, 255, thickness, cv2.LINE_AA)
        rotated = cv2.rotate(canvas, cv2.ROTATE_90_COUNTERCLOCKWISE)
        r_h, r_w = rotated.shape
        # Place vertical label above the baseline with a bit of gap so minus doesn't touch the ruler
        x0 = x - r_w // 2
        y0 = max(0, y_baseline - r_h - 10)
        x0 = max(0, min(bar_w - r_w, x0))
        y0 = max(0, min(h - r_h, y0))
        roi = bar[y0 : y0 + r_h, x0 : x0 + r_w]
        if roi.size > 0:
            r_h_roi, r_w_roi = roi.shape[0], roi.shape[1]
            mask = rotated[:r_h_roi, :r_w_roi] > 128
            if mask.any():
                for c in range(3):
                    roi[:, :, c][mask] = DB_LABEL_COLOR[c]


def draw_spectrum_analyzer(
    frame: np.ndarray,
    fft_data: np.ndarray,
    f_axis: np.ndarray,
    f_min: float,
    f_max: float,
    freq_bar_width: int,
    f_display_max: float,
    enabled: bool = True,
) -> None:
    """
    Draw the spectrum analyzer bar on the RIGHT side of the frame.

    enabled=True  -> dB mode: dB-scaled bars + curve + bottom ruler.
    enabled=False -> NORM mode: normalized (power^0.4) bars + curve, no ruler.
    Bandpass overlay is always drawn.
    """
    h, w, _ = frame.shape
    bar_w = int(max(1, freq_bar_width))
    f_display_max = float(f_display_max) if float(f_display_max) > 0 else 1.0
    bar_left = w - bar_w
    bar_right = w

    mag2 = np.sum(np.abs(fft_data) ** 2, axis=0).real
    valid = f_axis <= f_display_max
    f_valid = f_axis[valid]
    mag_valid = mag2[valid]

    bar = _get_panel_bg(h, bar_w).copy()

    # In dB mode reserve space for ruler at bottom
    use_db = enabled
    if use_db:
        graph_h = h - RULER_HEIGHT
    else:
        graph_h = h
    graph_h_safe = max(1, graph_h)

    graph_left = BAR_MARGIN_LEFT
    graph_right = bar_w - BAR_MARGIN_RIGHT
    graph_width = max(2, graph_right - graph_left)

    if mag_valid.size > 0:
        if use_db:
            ref = float(np.max(mag_valid)) + 1e-20
            mag_db = 10.0 * np.log10((mag_valid.astype(np.float64) + 1e-20) / ref)
            db_span = SPECTRUM_DB_MAX - SPECTRUM_DB_MIN
            frac = (mag_db - SPECTRUM_DB_MIN) / db_span
            frac = np.clip(frac, 0.0, 1.0)
        else:
            mag_norm = mag_valid / (float(np.max(mag_valid)) + 1e-12)
            frac = (mag_norm ** 0.4).astype(np.float64)

        for f, f_val in zip(f_valid, frac):
            y = int(graph_h_safe - 1 - (float(f) / f_display_max) * (graph_h_safe - 1))
            y = int(np.clip(y, 0, graph_h_safe - 1))
            length = int(float(f_val) * graph_width)
            x0 = graph_right - length
            x1 = graph_right
            x0 = int(np.clip(x0, graph_left, graph_right))
            x1 = int(np.clip(x1, graph_left, graph_right))

            t = y / max(1, graph_h_safe - 1)
            if float(f_min) <= float(f) <= float(f_max):
                color = tuple(int((1 - t) * GRAPH_COLOR_TOP[c] + t * GRAPH_COLOR_BOT[c]) for c in range(3))
            else:
                color = tuple(int((1 - t) * GRAPH_DIM_TOP[c] + t * GRAPH_DIM_BOT[c]) for c in range(3))
            if length > 0:
                cv2.line(bar, (x0, y), (x1, y), color, 1)

        pts = []
        for f, f_val in zip(f_valid, frac):
            y = int(np.clip(graph_h_safe - 1 - (float(f) / f_display_max) * (graph_h_safe - 1), 0, graph_h_safe - 1))
            length = int(float(f_val) * graph_width)
            x = graph_right - length
            x = int(np.clip(x, graph_left, graph_right))
            pts.append((x, y))
        if len(pts) >= 2:
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.polylines(
                bar, [pts_arr], isClosed=False,
                color=SPECTRUM_CURVE_BGR, thickness=SPECTRUM_CURVE_THICKNESS, lineType=cv2.LINE_AA
            )

        if use_db:
            _draw_db_ruler_bottom(bar, bar_w, graph_h)

    # ---- Bandpass overlay (use full h so drag handling in main matches) ----
    y_min = int(np.clip(freq_to_y(f_min, h, f_display_max), 0, h - 1))
    y_max = int(np.clip(freq_to_y(f_max, h, f_display_max), 0, h - 1))
    label_x = BAR_MARGIN_LEFT
    fmin_khz = float(f_min) / 1000.0
    fmax_khz = float(f_max) / 1000.0
    y_min_txt = int(np.clip(y_min - 6, 12, h - 6))
    y_max_txt = int(np.clip(y_max - 6, 12, h - 6))

    y_top = min(y_min, y_max)
    y_bottom = max(y_min, y_max)
    if y_bottom > y_top:
        overlay = bar.copy()
        tint_blue = (240, 180, 70)
        cv2.rectangle(overlay, (1, y_top), (bar_w - 2, y_bottom), tint_blue, -1)
        cv2.addWeighted(overlay, 0.12, bar, 0.88, 0, bar)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for (txt, y_pos) in [(f"{fmin_khz:5.1f} kHz", y_min_txt), (f"{fmax_khz:5.1f} kHz", y_max_txt)]:
        cv2.putText(bar, txt, (label_x, y_pos), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.line(bar, (0, y_min), (bar_w - 1, y_min), FREQ_BAR_BLUE, 2)
    cv2.line(bar, (0, y_max), (bar_w - 1, y_max), FREQ_BAR_BLUE, 2)
    cv2.line(bar, (0, y_min), (0, y_max), FREQ_BAR_BLUE, 2)
    cv2.line(bar, (bar_w - 1, y_min), (bar_w - 1, y_max), FREQ_BAR_BLUE, 2)

    handle_x = bar_w // 2
    cv2.circle(bar, (handle_x, y_min), 7, FREQ_BAR_BLUE, -1, cv2.LINE_AA)
    cv2.circle(bar, (handle_x, y_max), 7, FREQ_BAR_BLUE, -1, cv2.LINE_AA)
    cv2.circle(bar, (handle_x, y_min), 7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(bar, (handle_x, y_max), 7, (255, 255, 255), 1, cv2.LINE_AA)

    frame[:, bar_left:bar_right, :] = bar
