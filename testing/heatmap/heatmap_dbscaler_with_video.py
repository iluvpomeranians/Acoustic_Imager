#!/usr/bin/env python3
"""
NEED TO DOUBLE CHECK THIS BEC I MIGHTVE MISSED SOME THINGS
Acoustic Imager - Fermat Spiral Heatmap (Interactive Bandpass Demo)

UPDATED:
- Live Raspberry Pi 5 camera feed used as background
- Solid-color background code is COMMENTED OUT (not removed)
"""

import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import cv2
from numpy.linalg import eigh, pinv, eig

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1] / "dataframe"))
from fftframe import FFTFrame  # type: ignore

from heatmap_pipeline_test import (  # type: ignore
    apply_heatmap_overlay,
    # create_background_frame,   # <<< COMMENTED OUT (solid color background)
)

# ===============================================================
# 1. Configuration
# ===============================================================
N_MICS = 16
SAMPLES_PER_CHANNEL = 1024
SAMPLE_RATE_HZ = 150000
SPEED_SOUND = 343.0
NOISE_POWER = 0.0005
WINDOW_NAME = "Fermat Heatmap + Bandpass (MUSIC)"

# Display config
WIDTH = 1024
HEIGHT = 600
FPS = 30
ALPHA = 0.7

# Right-side frequency bar width (pixels)
FREQ_BAR_WIDTH = 200

# Beamforming / scanning grid
ANGLES = np.linspace(-90, 90, 181)

# Deterministic simulated sources
SOURCE_FREQS = [9000, 11000, 30000]
SOURCE_ANGLES = [-35.0, 0.0, 40.0]
SOURCE_AMPLS = [0.6, 1.0, 2.0]
N_SOURCES = len(SOURCE_ANGLES)

# FFT frequency axis
f_axis = np.fft.rfftfreq(SAMPLES_PER_CHANNEL, 1 / SAMPLE_RATE_HZ)

F_DISPLAY_MAX = 45000.0
ABSOLUTE_MAX_POWER = 1e-12

# ===============================================================
# 2. Camera setup (Raspberry Pi 5 / libcamera)
# ===============================================================
CAMERA_INDEX = 0  # CSI camera is usually index 0

# ===============================================================
# 3. Geometry setup (Fermat spiral)
# ===============================================================
golden_angle = np.deg2rad(137.5)
aperture_radius = 0.010
c_geom = aperture_radius / np.sqrt(N_MICS - 1)

x_coords, y_coords = [], []
for n in range(N_MICS):
    r = c_geom * np.sqrt(n)
    theta = n * golden_angle
    x_coords.append(r * np.cos(theta))
    y_coords.append(r * np.sin(theta))

x_coords = np.array(x_coords)
y_coords = np.array(y_coords)

pitch = np.mean(
    np.diff(sorted(np.unique(np.sqrt(x_coords**2 + y_coords**2))))
)

# ===============================================================
# 4. Simulated FFT frame generator
# ===============================================================
def generate_fft_frame_from_dataframe(angle_degs: List[float]) -> FFTFrame:
    frame = FFTFrame()
    frame.channel_count = N_MICS
    frame.sampling_rate = SAMPLE_RATE_HZ
    frame.fft_size = SAMPLES_PER_CHANNEL
    frame.frame_id += 1

    t = np.arange(SAMPLES_PER_CHANNEL) / SAMPLE_RATE_HZ
    mic_signals = np.zeros((N_MICS, len(t)), dtype=np.float32)

    for src_idx, angle_deg in enumerate(angle_degs):
        angle_rad = np.deg2rad(angle_deg)
        f = SOURCE_FREQS[src_idx]
        amp = SOURCE_AMPLS[src_idx]

        for i in range(N_MICS):
            delay = -(x_coords[i] * np.cos(angle_rad) +
                      y_coords[i] * np.sin(angle_rad)) / SPEED_SOUND
            mic_signals[i] += amp * np.sin(2 * np.pi * f * (t - delay))

    mic_signals += np.random.normal(0, np.sqrt(NOISE_POWER), mic_signals.shape)
    fft_data = np.fft.rfft(mic_signals, axis=1)
    frame.fft_data = fft_data.astype(np.complex64)
    return frame

# ===============================================================
# 5. Beamforming (MUSIC / ESPRIT)
# ===============================================================
def music_spectrum(R, angles, f_signal, n_sources):
    eigvals, eigvecs = eigh(R)
    idx = eigvals.argsort()[::-1]
    En = eigvecs[:, idx[n_sources:]]

    spectrum = []
    for ang in angles:
        theta = np.deg2rad(ang)
        a = np.exp(
            -1j * 2 * np.pi * f_signal / SPEED_SOUND *
            -(x_coords * np.cos(theta) + y_coords * np.sin(theta))
        )[:, None]

        P = 1.0 / np.real(a.conj().T @ En @ En.conj().T @ a)
        spectrum.append(P[0, 0])

    spec = np.array(spectrum)
    return spec / (spec.max() + 1e-12)

def esprit_estimate(R, f_signal, n_sources):
    eigvals, eigvecs = eigh(R)
    idx = eigvals.argsort()[::-1]
    Es = eigvecs[:, idx[:n_sources]]
    Es1, Es2 = Es[:-1], Es[1:]
    phi = pinv(Es1) @ Es2
    eigs_phi, _ = eig(phi)
    psi = np.angle(eigs_phi)
    val = -(psi * SPEED_SOUND) / (2 * np.pi * f_signal * pitch)
    val = np.clip(np.real(val), -1.0, 1.0)
    return np.degrees(np.arcsin(val))

# ===============================================================
# 6. Heatmap + frequency bar helpers (UNCHANGED)
# ===============================================================
def spectra_to_heatmap_absolute(spec_matrix, power_per_source,
                                out_width, out_height,
                                db_min=-40.0, db_max=0.0):

    Nsrc, Nang = spec_matrix.shape
    power_abs = np.maximum(power_per_source, 1e-12)
    power_db = 10 * np.log10(power_abs)

    power_norm = (power_db - db_min) / (db_max - db_min)
    power_norm = np.clip(power_norm, 0.0, 1.0)

    peak_indices = []
    sharpness = []

    for i in range(Nsrc):
        row = spec_matrix[i]
        idx = np.argmax(row)
        peak_indices.append(idx)

        left = row[idx-1] if idx > 0 else row[idx]
        right = row[idx+1] if idx < Nang-1 else row[idx]
        sharpness.append(max(row[idx] - 0.5*(left + right), 1e-12))

    sharpness = np.array(sharpness)
    sharpness /= sharpness.max() + 1e-12

    h, w = out_height, out_width
    heatmap = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    def ang_to_px(idx): return int(idx / Nang * w)
    base_radius = 60

    for i in range(Nsrc):
        cx = ang_to_px(peak_indices[i])
        cy = h // 2
        sigma = (base_radius * (0.7 + 0.3 * sharpness[i])) / 1.8
        amp = power_norm[i]

        heatmap += amp * np.exp(
            -((xx - cx)**2 + (yy - cy)**2) / (2*sigma*sigma)
        )

    heatmap = np.clip(heatmap, 0.0, 1.0)
    return (heatmap * 255).astype(np.uint8)

def draw_frequency_bar(frame, fft_data, f_axis, f_min, f_max):
    h, w, _ = frame.shape
    bar_w = FREQ_BAR_WIDTH
    left = w - bar_w

    mag2 = np.sum(np.abs(fft_data)**2, axis=0).real
    valid = f_axis <= F_DISPLAY_MAX

    f_valid = f_axis[valid]
    mag_valid = mag2[valid]

    mag_norm = mag_valid / (mag_valid.max() + 1e-12)
    mag_norm = mag_norm ** 0.4

    bar = np.zeros((h, bar_w, 3), dtype=np.uint8)

    for f, m in zip(f_valid, mag_norm):
        y = int(h - 1 - (f / F_DISPLAY_MAX) * (h - 1))
        length = int(m * (bar_w - 20))
        x0, x1 = bar_w - 5 - length, bar_w - 5

        color = (0,255,255) if f_min <= f <= f_max else (120,120,255)
        if length > 0:
            cv2.line(bar, (x0, y), (x1, y), color, 1)

    frame[:, left:] = bar

def draw_db_colorbar(frame, db_min, db_max, width=50):
    h = frame.shape[0]
    bar = np.zeros((h, width), dtype=np.uint8)

    for y in range(h):
        bar[h - 1 - y] = int((y / (h - 1)) * 255)

    bar_color = cv2.applyColorMap(bar, cv2.COLORMAP_JET)
    frame[:, :width] = bar_color

    cv2.putText(frame, f"{db_max:.0f} dB", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, f"{db_min:.0f} dB", (5, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

# ===============================================================
# 7. UI helpers
# ===============================================================
def nothing(x): pass

CURSOR_POS = (0, 0)
def mouse_move(event, x, y, flags, param):
    global CURSOR_POS
    if event == cv2.EVENT_MOUSEMOVE:
        CURSOR_POS = (x, y)

# ===============================================================
# 8. MAIN LOOP
# ===============================================================
def main():
    print("Acoustic Imager - Live Camera Mode")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, mouse_move)

    cv2.createTrackbar("f_min_kHz", WINDOW_NAME, 0, 45, nothing)
    cv2.createTrackbar("f_max_kHz", WINDOW_NAME, 45, 45, nothing)

    # ---------------- Camera init ----------------
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Could not open Raspberry Pi camera")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, cam_frame = cap.read()
            if not ret:
                continue

            cam_frame = cv2.resize(cam_frame, (WIDTH, HEIGHT))
            background = cam_frame.copy()

            # Animate sources (simulation)
            for k in range(N_SOURCES):
                SOURCE_ANGLES[k] += (0.15 + 0.05 * k)
                if SOURCE_ANGLES[k] > 90:
                    SOURCE_ANGLES[k] = -90

            frame = generate_fft_frame_from_dataframe(SOURCE_ANGLES)

            f_min = cv2.getTrackbarPos("f_min_kHz", WINDOW_NAME) * 1000.0
            f_max = cv2.getTrackbarPos("f_max_kHz", WINDOW_NAME) * 1000.0
            f_max = max(f_max, f_min)

            left_width = WIDTH - FREQ_BAR_WIDTH
            selected = [i for i,f in enumerate(SOURCE_FREQS) if f_min <= f <= f_max]

            if selected:
                spec_matrix = []
                powers = []

                for idx in selected:
                    f_sig = SOURCE_FREQS[idx]
                    f_idx = np.argmin(np.abs(f_axis - f_sig))
                    Xf = frame.fft_data[:, f_idx][:, None]
                    R = Xf @ Xf.conj().T

                    spec_matrix.append(music_spectrum(R, ANGLES, f_sig, len(selected)))
                    powers.append(np.sum(np.abs(Xf)**2).real)

                heatmap = spectra_to_heatmap_absolute(
                    np.array(spec_matrix),
                    np.array(powers),
                    left_width, HEIGHT
                )
            else:
                heatmap = np.zeros((HEIGHT, left_width), dtype=np.uint8)

            left_bg = background[:, :left_width]
            background[:, :left_width] = apply_heatmap_overlay(
                heatmap, left_bg, ALPHA
            )

            draw_frequency_bar(background, frame.fft_data, f_axis, f_min, f_max)
            draw_db_colorbar(background, -40, 0)

            cv2.imshow(WINDOW_NAME, background)

            key = cv2.waitKey(int(1000 // FPS)) & 0xFF
            if key in (ord('q'), 27):
                break

            frame_count += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

# ===============================================================
if __name__ == "__main__":
    main()
