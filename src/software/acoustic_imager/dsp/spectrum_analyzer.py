"""
Spectrum analyzer overlay: frequency bar with smooth spectrum curve.

Three modes (MENU: SPECTRUM: dB / NORM / LITE):
- dB: amplitude in dB (relative to peak), bottom ruler, spectrum curve, bandpass overlay.
- NORM: normalized amplitude (power^0.4), spectrum curve, bandpass overlay.
- LITE: normalized bars + bandpass overlay only (no spectrum curve; lighter draw for performance).

Uses the same FFT data as the beamforming pipeline.
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

# Subsample curve to reduce polylines cost (step 2 = every 2nd bin)
CURVE_SUBSAMPLE = 2

# Cache for dB ruler strip (bar_w, h) -> (RULER_HEIGHT, bar_w, 3); panel gradient depends on h
_RULER_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _get_cached_db_ruler(bar_w: int, full_bar_bg: np.ndarray) -> np.ndarray:
    """Return the bottom ruler strip (RULER_HEIGHT x bar_w), drawing once per (bar_w, h) and caching."""
    h = full_bar_bg.shape[0]
    key = (bar_w, h)
    cached = _RULER_CACHE.get(key)
    if cached is not None:
        return cached
    h = full_bar_bg.shape[0]
    strip = full_bar_bg[h - RULER_HEIGHT : h, :].copy()
    graph_left = BAR_MARGIN_LEFT
    graph_right = bar_w - BAR_MARGIN_RIGHT
    graph_width = max(1, graph_right - graph_left)
    db_span = SPECTRUM_DB_MAX - SPECTRUM_DB_MIN
    strip_h = strip.shape[0]
    y_baseline = strip_h - 1
    y_tick_top = y_baseline - 8
    y_tick_bottom = y_baseline
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    thickness = 1

    cv2.line(strip, (graph_left, y_baseline), (graph_right, y_baseline), RULER_TICK_COLOR, 1, cv2.LINE_AA)
    for db in range(int(SPECTRUM_DB_MIN), int(SPECTRUM_DB_MAX) + 1, 10):
        frac = (db - SPECTRUM_DB_MIN) / db_span
        x = int(np.clip(graph_left + frac * graph_width, graph_left, graph_right))
        cv2.line(strip, (x, y_tick_top), (x, y_tick_bottom), RULER_TICK_COLOR, 1, cv2.LINE_AA)
        label = f"{db}" if db != 0 else "0"
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        pad = 2
        canvas = np.zeros((th + 2 * pad, tw + 2 * pad), dtype=np.uint8)
        cv2.putText(canvas, label, (pad, pad + th), font, font_scale, 255, thickness, cv2.LINE_AA)
        rotated = cv2.rotate(canvas, cv2.ROTATE_90_COUNTERCLOCKWISE)
        r_h, r_w = rotated.shape
        x0 = max(0, min(bar_w - r_w, x - r_w // 2))
        y0 = max(0, min(strip_h - r_h, y_baseline - r_h - 10))
        roi = strip[y0 : y0 + r_h, x0 : x0 + r_w]
        if roi.size > 0:
            r_h_roi, r_w_roi = roi.shape[0], roi.shape[1]
            mask = rotated[:r_h_roi, :r_w_roi] > 128
            if mask.any():
                for c in range(3):
                    roi[:, :, c][mask] = DB_LABEL_COLOR[c]
    _RULER_CACHE[key] = strip
    return strip


def draw_spectrum_analyzer(
    frame: np.ndarray,
    fft_data: np.ndarray,
    f_axis: np.ndarray,
    f_min: float,
    f_max: float,
    freq_bar_width: int,
    f_display_max: float,
    mode: str = "dB",
) -> None:
    """
    Draw the spectrum analyzer bar on the RIGHT side of the frame.

    mode="dB"   -> dB-scaled bars + curve + bottom ruler + bandpass overlay.
    mode="NORM" -> normalized (power^0.4) bars + curve + bandpass overlay.
    mode="LITE" -> normalized bars + bandpass overlay only (no spectrum curve; lighter draw for performance).
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

    panel_bg = _get_panel_bg(h, bar_w)
    bar = panel_bg.copy()

    use_db = mode == "dB"
    draw_curve = mode != "LITE"   # LITE skips the continuous spectrum curve (saves FPS)
    draw_bandpass = True         # all modes show bandpass (sliding window)
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

        # Vectorized bar drawing: one segment per row (max length per row), slice assign
        y_float = (graph_h_safe - 1) - (f_valid.astype(np.float64) / f_display_max) * (graph_h_safe - 1)
        y_int = np.clip(y_float.astype(np.int32), 0, graph_h_safe - 1)
        length_all = (frac * graph_width).astype(np.int32)
        length_all = np.clip(length_all, 0, graph_width)
        in_band = (f_valid >= float(f_min)) & (f_valid <= float(f_max))
        t_row = np.arange(graph_h_safe, dtype=np.float64) / max(1, graph_h_safe - 1)
        color_top = np.array(GRAPH_COLOR_TOP, dtype=np.float64)
        color_bot = np.array(GRAPH_COLOR_BOT, dtype=np.float64)
        dim_top = np.array(GRAPH_DIM_TOP, dtype=np.float64)
        dim_bot = np.array(GRAPH_DIM_BOT, dtype=np.float64)
        length_per_row = np.zeros(graph_h_safe, dtype=np.int32)
        in_band_per_row = np.zeros(graph_h_safe, dtype=bool)
        for i in range(len(y_int)):
            r = y_int[i]
            if length_all[i] >= length_per_row[r]:
                length_per_row[r] = length_all[i]
                in_band_per_row[r] = in_band[i]
        for row in range(graph_h_safe):
            L = length_per_row[row]
            if L <= 0:
                continue
            x0 = int(np.clip(graph_right - L, graph_left, graph_right))
            x1 = graph_right
            t = t_row[row]
            if in_band_per_row[row]:
                color = ((1 - t) * color_top + t * color_bot).astype(np.uint8)
            else:
                color = ((1 - t) * dim_top + t * dim_bot).astype(np.uint8)
            bar[row, x0:x1, :] = color

        if draw_curve:
            step = max(1, CURVE_SUBSAMPLE)
            idx = np.arange(0, len(f_valid), step)
            if len(idx) < 2:
                idx = np.arange(len(f_valid))
            f_sub = f_valid[idx]
            frac_sub = frac[idx]
            y_curve = np.clip(
                (graph_h_safe - 1) - (f_sub.astype(np.float64) / f_display_max) * (graph_h_safe - 1),
                0, graph_h_safe - 1
            ).astype(np.int32)
            length_curve = (frac_sub * graph_width).astype(np.int32)
            x_curve = np.clip(graph_right - length_curve, graph_left, graph_right).astype(np.int32)
            pts_arr = np.column_stack((x_curve, y_curve)).astype(np.int32)
            if len(pts_arr) >= 2:
                cv2.polylines(
                    bar, [pts_arr], isClosed=False,
                    color=SPECTRUM_CURVE_BGR, thickness=SPECTRUM_CURVE_THICKNESS, lineType=cv2.LINE_AA
                )

        if use_db:
            ruler_strip = _get_cached_db_ruler(bar_w, panel_bg)
            bar[graph_h : h, :] = ruler_strip

    # ---- Bandpass overlay (sliding window; all modes) ----
    if draw_bandpass:
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
            tint_blue = (240, 180, 70)
            roi = bar[y_top : y_bottom + 1, 1 : bar_w - 1]
            overlay_roi = np.full((roi.shape[0], roi.shape[1], 3), tint_blue, dtype=roi.dtype)
            blended = np.empty_like(roi)
            cv2.addWeighted(overlay_roi, 0.12, roi, 0.88, 0, blended)
            np.copyto(roi, blended)

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
