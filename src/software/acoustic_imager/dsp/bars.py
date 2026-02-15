# dsp/bars.py
from __future__ import annotations

import numpy as np
import cv2


def freq_to_y(freq_hz: float, h: int, f_display_max: float) -> int:
    """Map frequency (Hz) -> y pixel (0..h-1)."""
    h = int(h)
    if h <= 1:
        return 0
    f_display_max = float(f_display_max) if float(f_display_max) > 0 else 1.0
    freq_hz = float(np.clip(freq_hz, 0.0, f_display_max))
    return int(h - 1 - (freq_hz / f_display_max) * (h - 1))


def y_to_freq(y: int, h: int, f_display_max: float) -> float:
    """Map y pixel (0..h-1) -> frequency (Hz)."""
    h = int(h)
    if h <= 1:
        return 0.0
    f_display_max = float(f_display_max) if float(f_display_max) > 0 else 1.0
    y = int(np.clip(y, 0, h - 1))
    frac = 1.0 - (y / (h - 1))
    return float(np.clip(frac * f_display_max, 0.0, f_display_max))


def draw_frequency_bar(
    frame: np.ndarray,
    fft_data: np.ndarray,
    f_axis: np.ndarray,
    f_min: float,
    f_max: float,
    freq_bar_width: int,
    f_display_max: float,
) -> None:
    """
    Draws the frequency bar on the RIGHT side of the frame, same as monolith.
    """
    h, w, _ = frame.shape

    bar_w = int(max(1, freq_bar_width))
    f_display_max = float(f_display_max) if float(f_display_max) > 0 else 1.0

    bar_left = w - bar_w
    bar_right = w

    # power per bin (sum across mics)
    mag2 = np.sum(np.abs(fft_data) ** 2, axis=0).real

    valid = f_axis <= f_display_max
    f_valid = f_axis[valid]
    mag_valid = mag2[valid]

    bar = np.zeros((h, bar_w, 3), dtype=np.uint8)

    if mag_valid.size > 0:
        mag_norm = mag_valid / (mag_valid.max() + 1e-12)
        mag_norm = mag_norm ** 0.4

        for f, m in zip(f_valid, mag_norm):
            y = int(h - 1 - (float(f) / f_display_max) * (h - 1))
            y = int(np.clip(y, 0, h - 1))

            length = int(float(m) * (bar_w - 20))
            x0 = bar_w - 5 - length
            x1 = bar_w - 5

            color = (0, 255, 255) if (float(f_min) <= float(f) <= float(f_max)) else (120, 120, 255)
            if length > 0:
                cv2.line(bar, (x0, y), (x1, y), color, 1)

    # band edges
    y_min = int(np.clip(freq_to_y(f_min, h, f_display_max), 0, h - 1))
    y_max = int(np.clip(freq_to_y(f_max, h, f_display_max), 0, h - 1))

    # labels
    label_x = 8
    fmin_khz = float(f_min) / 1000.0
    fmax_khz = float(f_max) / 1000.0
    y_min_txt = int(np.clip(y_min - 6, 12, h - 6))
    y_max_txt = int(np.clip(y_max - 6, 12, h - 6))

    cv2.putText(bar, f"{fmin_khz:5.1f} kHz", (label_x, y_min_txt),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(bar, f"{fmax_khz:5.1f} kHz", (label_x, y_max_txt),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.line(bar, (0, y_min), (bar_w - 1, y_min), (0, 255, 0), 1)
    cv2.line(bar, (0, y_max), (bar_w - 1, y_max), (0, 255, 0), 1)

    cv2.putText(bar, "Freq", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(bar, f"{int(f_display_max/1000)} kHz", (5, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(bar, "0", (5, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # draggable handles
    handle_x = bar_w // 2
    cv2.circle(bar, (handle_x, y_min), 7, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(bar, (handle_x, y_max), 7, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(bar, (handle_x, y_min), 7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(bar, (handle_x, y_max), 7, (0, 0, 0), 1, cv2.LINE_AA)

    frame[:, bar_left:bar_right, :] = bar


def draw_db_colorbar(
    frame: np.ndarray,
    db_min: float,
    db_max: float,
    width: int,
) -> None:
    """Draws the magma dB colorbar on the LEFT side of the frame."""
    h = int(frame.shape[0])
    width = int(max(1, width))

    # vectorized gradient (top=255, bottom=0 after flip)
    grad = np.linspace(0, 255, h, dtype=np.uint8)[::-1]
    bar = np.repeat(grad[:, None], width, axis=1)

    bar_color = cv2.applyColorMap(bar, cv2.COLORMAP_MAGMA)
    frame[:, :width] = bar_color

    cv2.putText(frame, f"{float(db_max):.0f} dB", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(frame, f"{float(db_min):.0f} dB", (0, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
