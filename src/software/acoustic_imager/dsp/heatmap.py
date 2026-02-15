# dsp/heatmap.py
from __future__ import annotations

import numpy as np
import cv2


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
) -> np.ndarray:
    """
    Drops in your FAST blend block but packaged as a function.

    Returns output_frame view (same object as base_frame).
    """
    left_bg = base_frame[:, :left_width, :]
    colored = cv2.applyColorMap(heatmap_left, cv2.COLORMAP_MAGMA)

    w = cv2.LUT(heatmap_left, w_lut_u8)
    w3 = cv2.merge((w, w, w))
    inv = cv2.bitwise_not(w3)

    c1 = cv2.multiply(colored, w3, scale=1 / 255.0)
    c2 = cv2.multiply(left_bg, inv, scale=1 / 255.0)
    left_out = cv2.add(c1, c2)

    base_frame[:, :left_width, :] = left_out
    return base_frame
