"""
Spectrum analyzer overlay: frequency bar with smooth spectrum curve.

Three modes (MENU: SPECTRUM: dB / NORM / LITE):
- dB: amplitude in dB (relative to peak), bottom ruler, spectrum curve, bandpass overlay.
- NORM: normalized amplitude (power^0.4), spectrum curve, bandpass overlay.
- LITE: normalized bars + bandpass overlay only (no spectrum curve; lighter draw for performance).

Uses the same FFT data as the beamforming pipeline.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import cv2

from ..config import (
    FREQ_BAR_BLUE,
    REL_DB_MIN,
    REL_DB_MAX,
)
from .bars import _get_panel_bg, freq_to_y, y_to_freq


SPECTRUM_DB_MIN = REL_DB_MIN
SPECTRUM_DB_MAX = REL_DB_MAX

BAR_MARGIN_LEFT = 8
BAR_MARGIN_RIGHT = 5
RULER_HEIGHT = 58   # bottom strip for dB ruler; tall enough for rotated labels at font_scale 0.40

GRAPH_COLOR_TOP = (0, 220, 255)
GRAPH_COLOR_BOT = (0, 120, 255)
GRAPH_DIM_TOP = (140, 160, 180)
GRAPH_DIM_BOT = (90, 120, 150)
SPECTRUM_CURVE_BGR = (255, 255, 200)
SPECTRUM_CURVE_THICKNESS = 2
DB_LABEL_COLOR = (200, 200, 200)
RULER_TICK_COLOR = (180, 180, 180)
# Numbers in line with each vertical tick, but nudge left so there's a gap (minus doesn't overlap the line)
RULER_LABEL_GAP_FROM_TICK = 5

# Subsample curve to reduce polylines cost (step 2 = every 2nd bin)
CURVE_SUBSAMPLE = 2

# Cache for dB ruler strip (bar_w, h) -> (RULER_HEIGHT, bar_w, 3) [kept for any non-overlay use]
_RULER_CACHE: dict[tuple[int, int], np.ndarray] = {}
# Cache for rotated dB label images (label_str -> rotated uint8 canvas, 255 = text)
_RULER_LABEL_CACHE: dict[str, np.ndarray] = {}

_RULER_FONT = cv2.FONT_HERSHEY_DUPLEX
_RULER_FONT_SCALE = 0.40
_RULER_THICKNESS = 1


def _get_rotated_label_image(label: str) -> np.ndarray:
    """Return rotated single-channel canvas (255 where text) for the given label. Cached."""
    if label in _RULER_LABEL_CACHE:
        return _RULER_LABEL_CACHE[label]
    (tw, th), _ = cv2.getTextSize(label, _RULER_FONT, _RULER_FONT_SCALE, _RULER_THICKNESS)
    pad = 2
    canvas = np.zeros((th + 2 * pad, tw + 2 * pad), dtype=np.uint8)
    cv2.putText(canvas, label, (pad, pad + th), _RULER_FONT, _RULER_FONT_SCALE, 255, _RULER_THICKNESS, cv2.LINE_AA)
    rotated = cv2.rotate(canvas, cv2.ROTATE_90_COUNTERCLOCKWISE)
    _RULER_LABEL_CACHE[label] = rotated
    return rotated


def _draw_db_ruler_onto(roi: np.ndarray, bar_w: int) -> None:
    """Draw only ruler graphics (baseline, ticks, labels) onto roi. No background - graph shows through."""
    graph_left = BAR_MARGIN_LEFT
    graph_right = bar_w - BAR_MARGIN_RIGHT
    graph_width = max(1, graph_right - graph_left)
    db_span = SPECTRUM_DB_MAX - SPECTRUM_DB_MIN
    strip_h = roi.shape[0]
    y_baseline = strip_h - 1
    y_tick_top = y_baseline - 8
    y_tick_bottom = y_baseline

    cv2.line(roi, (graph_left, y_baseline), (graph_right, y_baseline), RULER_TICK_COLOR, 1, cv2.LINE_AA)
    # Ruler: left = 0 dB (reference/loud), right = -60 dB (quiet), to match curve (peaks bulge left)
    for db in range(int(SPECTRUM_DB_MIN), int(SPECTRUM_DB_MAX) + 1, 10):
        frac = (SPECTRUM_DB_MAX - db) / db_span
        x = int(np.clip(graph_left + frac * graph_width, graph_left, graph_right))
        cv2.line(roi, (x, y_tick_top), (x, y_tick_bottom), RULER_TICK_COLOR, 1, cv2.LINE_AA)
        label = f"{db} dB" if db != 0 else "0 dB"
        rotated = _get_rotated_label_image(label)
        r_h, r_w = rotated.shape
        x0 = x - r_w // 2 - RULER_LABEL_GAP_FROM_TICK
        x0 = max(0, min(bar_w - r_w, x0))
        y0 = max(0, min(strip_h - r_h, y_baseline - r_h - 10))
        r_h_clip = min(r_h, strip_h - y0)
        r_w_clip = min(r_w, bar_w - x0)
        if r_h_clip > 0 and r_w_clip > 0:
            label_roi = roi[y0 : y0 + r_h_clip, x0 : x0 + r_w_clip]
            mask = rotated[:r_h_clip, :r_w_clip] > 128
            if mask.any():
                for c in range(3):
                    label_roi[:, :, c][mask] = DB_LABEL_COLOR[c]


def _get_cached_db_ruler(bar_w: int, full_bar_bg: np.ndarray) -> np.ndarray:
    """Bottom horizontal ruler with background (legacy/cache). Prefer _draw_db_ruler_onto for transparent overlay."""
    h = full_bar_bg.shape[0]
    key = (bar_w, h)
    cached = _RULER_CACHE.get(key)
    if cached is not None and cached.shape[0] == RULER_HEIGHT:
        return cached
    if cached is not None:
        del _RULER_CACHE[key]
    strip = full_bar_bg[h - RULER_HEIGHT : h, :].copy()
    _draw_db_ruler_onto(strip, bar_w)
    _RULER_CACHE[key] = strip
    return strip


# Red vertical cursor for measuring freq/dB (drawn on top of bandpass)
SPECTRUM_CURSOR_COLOR_BGR = (0, 0, 255)


def spectrum_closest_curve_point(
    cursor_x_bar: float,
    tap_y: int,
    h: int,
    f_display_max: float,
    fft_data: np.ndarray,
    f_axis: np.ndarray,
    bar_w: int,
    use_db: bool = True,
) -> tuple[float, float]:
    """
    Find the point on the blue spectrum curve closest to (cursor_x_bar, tap_y).
    Returns (curve_x_bar, dot_freq_hz) so the dot always lands on the curve.
    """
    mag2 = np.sum(np.abs(fft_data) ** 2, axis=0).real
    valid = f_axis <= f_display_max
    f_valid = f_axis[valid]
    mag_valid = mag2[valid]
    graph_left = BAR_MARGIN_LEFT
    graph_right = bar_w - BAR_MARGIN_RIGHT
    graph_width = max(1, graph_right - graph_left)
    graph_h_safe = max(1, h)
    if mag_valid.size == 0:
        return (float((graph_left + graph_right) // 2), 0.0)
    if use_db:
        ref = float(np.max(mag_valid)) + 1e-20
        mag_db = 10.0 * np.log10((mag_valid.astype(np.float64) + 1e-20) / ref)
        db_span = SPECTRUM_DB_MAX - SPECTRUM_DB_MIN
        frac = np.clip((mag_db - SPECTRUM_DB_MIN) / db_span, 0.0, 1.0)
    else:
        mag_norm = mag_valid / (float(np.max(mag_valid)) + 1e-12)
        frac = (mag_norm ** 0.4).astype(np.float64)
    x_curve = np.clip(graph_right - frac * graph_width, graph_left, graph_right).astype(np.float64)
    y_curve = (graph_h_safe - 1) - (f_valid.astype(np.float64) / f_display_max) * (graph_h_safe - 1)
    y_curve = np.clip(y_curve, 0, graph_h_safe - 1)
    dist_sq = (x_curve - cursor_x_bar) ** 2 + (y_curve - float(tap_y)) ** 2
    idx = int(np.argmin(dist_sq))
    return (float(np.clip(x_curve[idx], graph_left, graph_right)), float(f_valid[idx]))


def spectrum_curve_x_at_y(
    tap_y: int,
    h: int,
    f_display_max: float,
    fft_data: np.ndarray,
    f_axis: np.ndarray,
    bar_w: int,
) -> float:
    """Return the curve x (bar coords) at the frequency corresponding to tap_y."""
    mag2 = np.sum(np.abs(fft_data) ** 2, axis=0).real
    valid = f_axis <= f_display_max
    f_valid = f_axis[valid]
    mag_valid = mag2[valid]
    if mag_valid.size == 0:
        return float(bar_w // 2)
    freq = y_to_freq(tap_y, h, f_display_max)
    idx = int(np.argmin(np.abs(f_valid - freq)))
    ref = float(np.max(mag_valid)) + 1e-20
    mag_db = 10.0 * np.log10((float(mag_valid[idx]) + 1e-20) / ref)
    db_span = SPECTRUM_DB_MAX - SPECTRUM_DB_MIN
    frac = float(np.clip((mag_db - SPECTRUM_DB_MIN) / db_span, 0.0, 1.0))
    graph_left = BAR_MARGIN_LEFT
    graph_right = bar_w - BAR_MARGIN_RIGHT
    graph_width = max(1, graph_right - graph_left)
    curve_x = graph_right - frac * graph_width
    return float(np.clip(curve_x, graph_left, graph_right))


def draw_spectrum_analyzer(
    frame: np.ndarray,
    fft_data: np.ndarray,
    f_axis: np.ndarray,
    f_min: float,
    f_max: float,
    freq_bar_width: int,
    f_display_max: float,
    mode: str = "dB",
    spectrum_cursor_x: Optional[float] = None,
    spectrum_cursor_dot_active: bool = False,
    spectrum_cursor_dot_freq: Optional[float] = None,
    spectrum_cursor_dot_dragging: bool = False,
    spectrum_cursor_dot_bar_pos: Optional[list] = None,
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
        graph_h = h - RULER_HEIGHT   # where ruler strip is placed (drawn on top)
    else:
        graph_h = h
    # Draw spectrum (bars + curve) at full height so graph isn't clipped; ruler is drawn on top of bottom strip
    draw_h = h
    graph_h_safe = max(1, draw_h)
    graph_left = BAR_MARGIN_LEFT
    graph_right = bar_w - BAR_MARGIN_RIGHT
    graph_width = max(2, graph_right - graph_left)

    # Cursor readout (freq/dB at vertical line); set inside mag_valid block, drawn after bandpass
    cursor_draw_x: Optional[int] = None
    dot_draw_x: Optional[int] = None  # when set, dot is drawn here (on curve) instead of on the line
    if spectrum_cursor_x is not None:
        cursor_draw_x = int(round(np.clip(spectrum_cursor_x, graph_left, graph_right)))
    cursor_draw_y: Optional[int] = None
    cursor_freq_hz: Optional[float] = None
    cursor_db_val: Optional[float] = None

    if mag_valid.size > 0:
        if use_db:
            ref = float(np.max(mag_valid)) + 1e-20
            mag_db = 10.0 * np.log10((mag_valid.astype(np.float64) + 1e-20) / ref)
            db_span = SPECTRUM_DB_MAX - SPECTRUM_DB_MIN
            frac = np.clip((mag_db - SPECTRUM_DB_MIN) / db_span, 0.0, 1.0)
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
            # Draw only ruler graphics (baseline, ticks, labels) on top of graph; no background so graph shows through
            _draw_db_ruler_onto(bar[graph_h : h, :], bar_w)

        # Compute cursor: line position; dot position/readout use pinned dot_freq when set (stops bouncing)
        if spectrum_cursor_x is not None:
            cx = float(np.clip(spectrum_cursor_x, graph_left, graph_right))
            cursor_draw_x = int(round(cx))
            if spectrum_cursor_dot_freq is not None and spectrum_cursor_dot_active:
                # Dot pinned to closest curve point: draw at (curve_x, curve_y) for this freq so it stays on the blue curve
                cursor_freq_hz = float(spectrum_cursor_dot_freq)
                idx = int(np.argmin(np.abs(f_valid - spectrum_cursor_dot_freq)))
                cursor_draw_y = int(np.clip((graph_h_safe - 1) - (cursor_freq_hz / f_display_max) * (graph_h_safe - 1), 0, graph_h_safe - 1))
                curve_x_at_dot = graph_right - frac[idx] * graph_width
                dot_draw_x = int(np.clip(round(curve_x_at_dot), graph_left, graph_right))
                cursor_mag = float(mag_valid[idx])
                ref_cursor = float(np.max(mag_valid)) + 1e-20
                cursor_db_val = 10.0 * np.log10((cursor_mag + 1e-20) / ref_cursor)
            else:
                # No pinned dot: dot follows curve at cursor x (for drag before first tap)
                # frac 0 = right (0 dB), frac 1 = left (-60 dB), so target_frac = (graph_right - cx) / width
                target_frac = (graph_right - cx) / max(1, graph_width)
                target_frac = float(np.clip(target_frac, 0.0, 1.0))
                idx = int(np.argmin(np.abs(frac - target_frac)))
                cursor_freq_hz = float(f_valid[idx])
                cursor_mag = float(mag_valid[idx])
                ref_cursor = float(np.max(mag_valid)) + 1e-20
                cursor_db_val = 10.0 * np.log10((cursor_mag + 1e-20) / ref_cursor)
                cursor_draw_y = int(np.clip((graph_h_safe - 1) - (cursor_freq_hz / f_display_max) * (graph_h_safe - 1), 0, graph_h_safe - 1))

    # ---- Bandpass overlay (sliding window; all modes) ----
    if draw_bandpass:
        # Use draw_h so overlay aligns with full-height graph (ruler is drawn on top of bottom strip)
        y_min = int(np.clip(freq_to_y(f_min, draw_h, f_display_max), 0, draw_h - 1))
        y_max = int(np.clip(freq_to_y(f_max, draw_h, f_display_max), 0, draw_h - 1))
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

    # ---- Spectrum cursor (red vertical line; dot + label only when user has tapped on line) ----
    if spectrum_cursor_dot_bar_pos is not None:
        spectrum_cursor_dot_bar_pos.clear()
    if cursor_draw_x is not None:
        cx = cursor_draw_x
        cv2.line(bar, (cx, 0), (cx, h - 1), SPECTRUM_CURSOR_COLOR_BGR, 2, cv2.LINE_AA)
    if (
        spectrum_cursor_dot_active
        and cursor_freq_hz is not None
        and cursor_db_val is not None
        and (dot_draw_x is not None or cursor_draw_x is not None)
    ):
        dx = dot_draw_x if dot_draw_x is not None else cursor_draw_x
        dot_r = 9 if spectrum_cursor_dot_dragging else 6
        cv2.circle(bar, (dx, cursor_draw_y), dot_r, SPECTRUM_CURSOR_COLOR_BGR, -1, cv2.LINE_AA)
        cv2.circle(bar, (dx, cursor_draw_y), dot_r, (255, 255, 255), 1, cv2.LINE_AA)
        if spectrum_cursor_dot_bar_pos is not None:
            spectrum_cursor_dot_bar_pos[:] = [float(dx), float(cursor_draw_y)]
        freq_khz = cursor_freq_hz / 1000.0
        # Show dB in same direction as ruler: ruler maps left = 0 dB, right = -60 dB (matches curve)
        db_span = SPECTRUM_DB_MAX - SPECTRUM_DB_MIN
        label_db = SPECTRUM_DB_MAX - (dx - graph_left) / max(1, graph_width) * db_span
        label_db = float(np.clip(label_db, SPECTRUM_DB_MIN, SPECTRUM_DB_MAX))
        label = f"{freq_khz:.1f} kHz  {label_db:.0f} dB"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        # Place label above the dot, avoid clipping
        ly = max(th + 2, cursor_draw_y - 8)
        lx = dx + 6
        if lx + tw > bar_w - 2:
            lx = dx - tw - 6
        lx = max(2, min(lx, bar_w - tw - 2))
        cv2.putText(bar, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    frame[:, bar_left:bar_right, :] = bar
