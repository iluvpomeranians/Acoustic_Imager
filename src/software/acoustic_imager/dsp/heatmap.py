# dsp/heatmap.py
from __future__ import annotations

from typing import Optional

import numpy as np
import cv2

# Colormap mapping
COLORMAP_DICT = {
    "MAGMA": cv2.COLORMAP_MAGMA,
    "JET": cv2.COLORMAP_JET,
    "TURBO": cv2.COLORMAP_TURBO,
    "INFERNO": cv2.COLORMAP_INFERNO,
}


def spectra_to_heatmap_absolute(
    spec_matrix: np.ndarray,
    power_rel: np.ndarray,
    out_width: int,
    out_height: int,
    db_min: float,
    db_max: float,
) -> np.ndarray:
    """
    SAME signature as your monolith.

    Faster heatmap builder:
      - Finds peak angle per source
      - Creates Gaussian blobs only in a small ROI (not full-frame meshgrid)
      - Accumulates in float32 heatmap, returns uint8 0..255
    """
    Nsrc, Nang = spec_matrix.shape
    if Nsrc == 0 or Nang == 0:
        return np.zeros((out_height, out_width), dtype=np.uint8)

    power_rel = np.maximum(power_rel.astype(np.float32), 1e-12)
    power_db = 10.0 * np.log10(power_rel)  # <= 0
    power_norm = (power_db - float(db_min)) / (float(db_max) - float(db_min) + 1e-12)
    power_norm = np.clip(power_norm, 0.0, 1.0)

    peak_idx = np.argmax(spec_matrix, axis=1).astype(np.int32)

    sharp = np.empty(Nsrc, dtype=np.float32)
    for i in range(Nsrc):
        idx = int(peak_idx[i])
        p = float(spec_matrix[i, idx])
        left = float(spec_matrix[i, idx - 1]) if idx > 0 else p
        right = float(spec_matrix[i, idx + 1]) if idx < (Nang - 1) else p
        sharp[i] = max(p - 0.5 * (left + right), 1e-12)

    sharp /= (sharp.max() + 1e-12)

    h, w = int(out_height), int(out_width)
    heat = np.zeros((h, w), dtype=np.float32)

    cx_all = (peak_idx.astype(np.float32) * (w - 1) / max(1, (Nang - 1))).astype(np.int32)
    cx_all = np.clip(cx_all, 0, w - 1)
    cy = h // 2

    base_radius = 60.0

    for i in range(Nsrc):
        amp = float(power_norm[i])
        if amp <= 0.0:
            continue

        cx = int(cx_all[i])

        blob_radius = base_radius * (0.7 + 0.3 * float(sharp[i]))
        sigma = blob_radius / 1.8
        sigma2 = float(2.0 * sigma * sigma + 1e-12)

        r = int(max(6, min(200, round(7.0 * sigma))))
        x0 = max(0, cx - r)
        x1 = min(w, cx + r + 1)
        y0 = max(0, cy - r)
        y1 = min(h, cy + r + 1)

        xs = (np.arange(x0, x1, dtype=np.float32) - cx)
        ys = (np.arange(y0, y1, dtype=np.float32) - cy)

        gx = np.exp(-(xs * xs) / sigma2)
        gy = np.exp(-(ys * ys) / sigma2)

        blob = amp * (gy[:, None] * gx[None, :]).astype(np.float32)
        heat[y0:y1, x0:x1] += blob

    heat = np.clip(heat, 0.0, 1.0)
    return (heat * 255.0).astype(np.uint8)


def build_w_lut_u8(alpha: float, blend_gamma: float) -> np.ndarray:
    """
    Rebuilds your W_LUT_U8 exactly (so main can call this instead of global init).
    """
    x = np.arange(256, dtype=np.float32) / 255.0
    x = np.power(x, float(blend_gamma)) * float(alpha)
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


def blend_heatmap_left(
    base_frame: np.ndarray,
    heatmap_left: np.ndarray,
    left_width: int,
    w_lut_u8: np.ndarray,
    colormap: str = "MAGMA",
) -> np.ndarray:
    """
    Drops in your FAST blend block but packaged as a function.

    Returns output_frame view (same object as base_frame).
    """
    left_bg = base_frame[:, :left_width, :]
    colormap_cv = COLORMAP_DICT.get(colormap, cv2.COLORMAP_MAGMA)
    colored = cv2.applyColorMap(heatmap_left, colormap_cv)

    w = cv2.LUT(heatmap_left, w_lut_u8)
    w3 = cv2.merge((w, w, w))
    inv = cv2.bitwise_not(w3)

    c1 = cv2.multiply(colored, w3, scale=1 / 255.0)
    c2 = cv2.multiply(left_bg, inv, scale=1 / 255.0)
    left_out = cv2.add(c1, c2)

    base_frame[:, :left_width, :] = left_out
    return base_frame


# Crosshairs: 5mm cross (two lines 5mm each), ~19 px at 96 DPI
CROSSHAIR_LENGTH_PX = 19
CROSSHAIR_HALF = CROSSHAIR_LENGTH_PX // 2
CROSSHAIR_COLOR = (220, 220, 220)
CROSSHAIR_THICKNESS = 1
CROSSHAIR_TOOLTIP_BG = (40, 40, 40)
CROSSHAIR_TOOLTIP_TEXT = (255, 255, 255)
CROSSHAIR_TRACK_RADIUS = 15
CROSSHAIR_DISMISS_RADIUS_PX = 25


def find_local_max(heatmap: np.ndarray, cx: int, cy: int, radius: int) -> tuple[int, int]:
    """Return (x, y) of pixel with max value in heatmap within radius of (cx, cy)."""
    h, w = heatmap.shape[:2]
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)
    roi = heatmap[y0:y1, x0:x1]
    if roi.size == 0:
        return (cx, cy)
    flat_idx = np.argmax(roi)
    iy = flat_idx // roi.shape[1]
    ix = flat_idx % roi.shape[1]
    return (x0 + ix, y0 + iy)


# Trend/acceleration colors (BGR): green = rising, red = falling, white = stable
TREND_COLOR_POS = (0, 200, 100)    # green
TREND_COLOR_NEG = (0, 100, 255)    # red
TREND_COLOR_ZERO = (200, 200, 200)  # light gray
TREND_COLOR_PENDING = (120, 120, 120)  # dim gray for placeholder (waiting for first avg)


def _trend_color(db: float, threshold: float = 0.3) -> tuple:
    """Return BGR color for trend/accel value."""
    if db is None:
        return CROSSHAIR_TOOLTIP_TEXT
    if db > threshold:
        return TREND_COLOR_POS
    if db < -threshold:
        return TREND_COLOR_NEG
    return TREND_COLOR_ZERO


def draw_crosshairs(
    frame: np.ndarray,
    center_x: int,
    center_y: int,
    left_width: int,
    height: int,
    heatmap_left: np.ndarray,
    rel_db_min: float,
    rel_db_max: float,
    f_peak_hz: float,
    trend_db: Optional[float] = None,
    accel_db: Optional[float] = None,
) -> None:
    """Draw 5mm cross and tooltip (dB, kHz, 3s trend, 12s accel in colors)."""
    if left_width <= 0 or height <= 0 or heatmap_left.size == 0:
        return
    cx = max(0, min(left_width - 1, int(center_x)))
    cy = max(0, min(height - 1, int(center_y)))
    x0 = max(0, cx - CROSSHAIR_HALF)
    x1 = min(left_width, cx + CROSSHAIR_HALF + 1)
    y0 = max(0, cy - CROSSHAIR_HALF)
    y1 = min(height, cy + CROSSHAIR_HALF + 1)
    cv2.line(frame, (x0, cy), (x1, cy), CROSSHAIR_COLOR, CROSSHAIR_THICKNESS, cv2.LINE_AA)
    cv2.line(frame, (cx, y0), (cx, y1), CROSSHAIR_COLOR, CROSSHAIR_THICKNESS, cv2.LINE_AA)

    val = int(heatmap_left[cy, cx])
    db = rel_db_min + (val / 255.0) * (rel_db_max - rel_db_min)
    f_peak_khz = f_peak_hz / 1000.0
    line1 = f"{db:+.0f} dB"
    line2 = f"{f_peak_khz:.1f} kHz"
    lines = [(line1, CROSSHAIR_TOOLTIP_TEXT), (line2, CROSSHAIR_TOOLTIP_TEXT)]
    # Always show 3s/12s slots; use placeholder until first avg is ready
    if trend_db is not None:
        lines.append((f"3s: {trend_db:+.1f} dB", _trend_color(trend_db)))
    else:
        lines.append(("3s: -- dB", TREND_COLOR_PENDING))
    if accel_db is not None:
        lines.append((f"12s: {accel_db:+.2f}/3s", _trend_color(accel_db, 0.05)))
    else:
        lines.append(("12s: --/3s", TREND_COLOR_PENDING))

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thick = 1
    pad = 6
    box_w = pad * 2
    box_h = pad * 2
    for txt, _ in lines:
        (tw, th), _ = cv2.getTextSize(txt, font, scale, thick)
        box_w = max(box_w, tw + 2 * pad)
        box_h += th + 2
    tx = cx + 14
    ty = cy - 4
    if tx + box_w > left_width:
        tx = cx - box_w - 14
    if ty < 0:
        ty = cy + 20
    if ty + box_h > height:
        ty = cy - box_h - 8
    tx = max(2, min(left_width - int(box_w) - 2, tx))
    ty = max(2, min(height - int(box_h) - 2, ty))
    cv2.rectangle(frame, (int(tx), int(ty)), (int(tx + box_w), int(ty + box_h)), CROSSHAIR_TOOLTIP_BG, -1)
    cv2.rectangle(frame, (int(tx), int(ty)), (int(tx + box_w), int(ty + box_h)), CROSSHAIR_COLOR, 1, cv2.LINE_AA)
    y_off = ty + pad
    for txt, color in lines:
        (_, th), _ = cv2.getTextSize(txt, font, scale, thick)
        cv2.putText(frame, txt, (int(tx + pad), int(y_off + th)), font, scale, color, thick, cv2.LINE_AA)
        y_off += th + 2
