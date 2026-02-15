# dsp/bars.py
from __future__ import annotations

import numpy as np
import cv2


def freq_to_y(freq_hz: float, h: int, f_display_max: float) -> int:
    """
    Same logic as your monolith, but f_display_max is passed in (no globals).
    """
    freq_hz = float(np.clip(freq_hz, 0.0, float(f_display_max)))
    return int(h - 1 - (freq_hz / float(f_display_max)) * (h - 1))


def y_to_freq(y: int, h: int, f_display_max: float) -> float:
    """
    Same logic as your monolith, but f_display_max is passed in (no globals).
    """
    y = int(np.clip(y, 0, h - 1))
    frac = 1.0 - (y / (h - 1))
    return float(np.clip(frac * float(f_display_max), 0.0, float(f_display_max)))


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
    Same drawing code as your monolith, but:
      - uses freq_bar_width instead of global FREQ_BAR_WIDTH
      - uses f_display_max instead of global F_DISPLAY_MAX
    """
    h, w, _ = frame.shape

    area_left = w - int(freq_bar_width)
    area_right = w

    bar_w = int(freq_bar_width)
    bar_left = area_left
    bar_right = area_right

    mag2 = np.sum(np.abs(fft_data) ** 2, axis=0).real

    valid = f_axis <= float(f_display_max)
    f_valid = f_axis[valid]
    mag_valid = mag2[valid]

    bar = np.zeros((h, bar_w, 3), dtype=np.uint8)
    if mag_valid.size > 0:
        mag_norm = mag_valid / (mag_valid.max() + 1e-12)
        mag_norm = mag_norm ** 0.4

        for f, m in zip(f_valid, mag_norm):
            y = int(h - 1 - (f / float(f_display_max)) * (h - 1))
            y = int(np.clip(y, 0, h - 1))
            length = int(m * (bar_w - 20))
            x0 = bar_w - 5 - length
            x1 = bar_w - 5
            color = (0, 255, 255) if (f_min <= f <= f_max) else (120, 120, 255)
            if length > 0:
                cv2.line(bar, (x0, y), (x1, y), color, 1)

    y_min = int(np.clip(freq_to_y(f_min, h, f_display_max), 0, h - 1))
    y_max = int(np.clip(freq_to_y(f_max, h, f_display_max), 0, h - 1))

    label_x = 8
    fmin_khz = f_min / 1000.0
    fmax_khz = f_max / 1000.0

    y_min_txt = int(np.clip(y_min - 6, 12, h - 6))
    y_max_txt = int(np.clip(y_max - 6, 12, h - 6))

    cv2.putText(bar, f"{fmin_khz:5.1f} kHz", (label_x, y_min_txt),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(bar, f"{fmax_khz:5.1f} kHz", (label_x, y_max_txt),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.line(bar, (0, y_min), (bar_w - 1, y_min), (0, 255, 0), 1)
    cv2.line(bar, (0, y_max), (bar_w - 1, y_max), (0, 255, 0), 1)

    cv2.putText(bar, "Freq", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(bar, f"{int(float(f_display_max)/1000)} kHz", (5, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(bar, "0", (5, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    handle_x = bar_w // 2
    cv2.circle(bar, (handle_x, int(y_min)), 7, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(bar, (handle_x, int(y_max)), 7, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(bar, (handle_x, int(y_min)), 7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(bar, (handle_x, int(y_max)), 7, (0, 0, 0), 1, cv2.LINE_AA)

    frame[:, bar_left:bar_right, :] = bar


def draw_db_colorbar(
    frame: np.ndarray,
    db_min: float,
    db_max: float,
    width: int,
) -> None:
    """
    Same as your monolith, but width is passed in (no global DB_BAR_WIDTH).
    """
    h = frame.shape[0]
    width = int(width)

    bar = np.zeros((h, width), dtype=np.uint8)
    for y in range(h):
        val = y / (h - 1)
        bar[h - 1 - y, :] = int(val * 255)

    bar_color = cv2.applyColorMap(bar, cv2.COLORMAP_MAGMA)
    frame[:, :width] = bar_color

    cv2.putText(frame, f"{db_max:.0f} dB", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(frame, f"{db_min:.0f} dB", (0, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
