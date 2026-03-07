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

# 180° protractor in tooltip (semicircle, flat base at bottom; light fill, dark outline)
PROTRACTOR_W = 76
PROTRACTOR_H = 60
PROTRACTOR_BUFFER_PX = 3   # gap from max(dB, kHz) text to protractor left edge
PROTRACTOR_FILL_BGR = (220, 200, 160)   # light blue-ish
PROTRACTOR_OUTLINE_BGR = (0, 0, 0)
PROTRACTOR_ARROW_BGR = (0, 0, 255)      # red arrow


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


def _draw_protractor_180(
    frame: np.ndarray,
    x0: int,
    y0: int,
    w: int,
    h: int,
    angle_deg: Optional[float],
) -> None:
    """Draw a 180° protractor (semicircle, flat base at bottom) with dynamic arrow and degree text.
    angle_deg: DOA in plane of array; 0° = normal (arrow up), +90° = right, -90° = left.
    """
    if w < 10 or h < 10:
        return
    cx = x0 + w // 2
    r = min((w - 4) // 2, (h - 6) // 2)
    if r < 4:
        return
    cy = y0 + r + 2   # center so flat base at y0 + 2*r + 2
    # Semicircle: top arc (OpenCV: 0=right, 90=down; top half = 180° to 360°)
    cv2.ellipse(frame, (cx, cy), (r, r), 0, 180, 360, PROTRACTOR_FILL_BGR, -1, cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), (r, r), 0, 180, 360, PROTRACTOR_OUTLINE_BGR, 1, cv2.LINE_AA)
    # Flat base
    cv2.line(frame, (cx - r, cy), (cx + r, cy), PROTRACTOR_OUTLINE_BGR, 1, cv2.LINE_AA)
    # Tick marks along arc (0°=left, 90°=top, 180°=right) and center dot
    for deg in [0, 30, 60, 90, 120, 150, 180]:
        # Circle: angle 0=right, 90=top; we want 0=left, 90=top, 180=right so use 180-deg
        a_rad = np.deg2rad(180 - deg)
        tx = cx + int((r - 2) * np.cos(a_rad))
        ty = cy - int((r - 2) * np.sin(a_rad))
        tx2 = cx + int(r * np.cos(a_rad))
        ty2 = cy - int(r * np.sin(a_rad))
        cv2.line(frame, (tx, ty), (tx2, ty2), PROTRACTOR_OUTLINE_BGR, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 2, PROTRACTOR_OUTLINE_BGR, -1, cv2.LINE_AA)
    # Dynamic arrow (0° = up, 90° = right, -90° = left)
    arrow_len = int(r * 0.65)
    if angle_deg is not None:
        rad = np.deg2rad(angle_deg)
        ex = int(cx + arrow_len * np.sin(rad))
        ey = int(cy - arrow_len * np.cos(rad))
        cv2.arrowedLine(frame, (cx, cy), (ex, ey), PROTRACTOR_ARROW_BGR, 2, cv2.LINE_AA, tipLength=0.25)
    # Degree text just below protractor base (one decimal; use " deg" - OpenCV default font doesn't render °)
    label = f"{angle_deg:.1f} deg" if angle_deg is not None else "-- deg"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    lx = cx - tw // 2
    ly = cy + 2 + th   # baseline 2px below flat base so text sits just under protractor
    cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.35, CROSSHAIR_TOOLTIP_TEXT, 1, cv2.LINE_AA)


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
    distance_to_source_m: Optional[float] = None,
    angle_deg: Optional[float] = None,
) -> None:
    """Draw 5mm cross and tooltip (dB, kHz, distance, 3s trend, 12s accel, 180° protractor)."""
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
    line3 = f"{distance_to_source_m:.2f} m" if distance_to_source_m is not None else "-- m"
    lines = [(line1, CROSSHAIR_TOOLTIP_TEXT), (line2, CROSSHAIR_TOOLTIP_TEXT), (line3, CROSSHAIR_TOOLTIP_TEXT)]
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
    # Extra height for horizontal bar between kHz and timed measurements (gap + line + gap)
    box_h += 7
    text_w = box_w
    # Protractor: 3px buffer from right edge of dB/kHz text (use max of first two line widths)
    (tw1, _), _ = cv2.getTextSize(lines[0][0], font, scale, thick)
    (tw2, _), _ = cv2.getTextSize(lines[1][0], font, scale, thick)
    max_db_khz_w = max(tw1, tw2)
    protractor_x0 = pad + max_db_khz_w + PROTRACTOR_BUFFER_PX
    box_w = max(text_w, protractor_x0 + PROTRACTOR_W + 2)
    box_h = max(box_h, PROTRACTOR_H + 4)
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
    # 180° protractor in top-right; left edge = 3px after end of dB/kHz text
    _draw_protractor_180(
        frame,
        int(tx + protractor_x0),
        int(ty + 2),
        PROTRACTOR_W,
        min(PROTRACTOR_H, int(box_h) - 4),
        angle_deg,
    )
    y_off = ty + pad
    for i, (txt, color) in enumerate(lines):
        (_, th), _ = cv2.getTextSize(txt, font, scale, thick)
        cv2.putText(frame, txt, (int(tx + pad), int(y_off + th)), font, scale, color, thick, cv2.LINE_AA)
        y_off += th + 2
        # Small horizontal bar between distance (index 2) and timed measurements (3s, 12s)
        if i == 2:
            bar_y = int(y_off + 2)
            bar_left = int(tx + pad)
            bar_right = int(tx + box_w - pad)
            cv2.line(frame, (bar_left, bar_y), (bar_right, bar_y), CROSSHAIR_COLOR, 1, cv2.LINE_AA)
            y_off = bar_y + 4
