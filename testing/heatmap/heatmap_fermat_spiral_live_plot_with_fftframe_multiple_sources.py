#!/usr/bin/env python3
"""
Acoustic Imager - Fermat Spiral Heatmap (Multi-Source, FFTFrame)

This test script combines:
  1) Deterministic multi-source plane-wave simulation on a Fermat spiral array
  2) FFTFrame-based STM32-style frame generation
  3) MUSIC DOA estimation across angles for multiple frequencies
  4) Heatmap rendering + overlay using the existing heatmap pipeline

Output:
  - Live OpenCV window with scrolling heatmap
  - Optional FFmpeg recording to MP4
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
# Reuse FFTFrame from ../dataframe
sys.path.append(str(Path(__file__).resolve().parents[1] / "dataframe"))
from fftframe import FFTFrame  # type: ignore

# Reuse heatmap helpers from this folder
from heatmap_pipeline_test import (  # type: ignore
    apply_heatmap_overlay,
    create_background_frame,
    setup_ffmpeg_process,
    cleanup_ffmpeg_process,
    send_to_ffmpeg,
)

# ===============================================================
# 1. Configuration
# ===============================================================
N_MICS = 16
SAMPLES_PER_CHANNEL = 1024
SAMPLE_RATE_HZ = 68001
SPEED_SOUND = 343.0
NOISE_POWER = 0.0005

# Heatmap / display config
WIDTH = 640
HEIGHT = 480
FPS = 30
ALPHA = 0.7
NUM_FRAMES = 600
OUTPUT_FILE = "heatmap_fermat_multisrc.mp4"
USE_FFMPEG = True  # set False if you don't want to record

# Beamforming / scanning grid
ANGLES = np.linspace(-90, 90, 181)  # 1° resolution

# Multiple deterministic sources
SOURCE_FREQS = [9000, 11000, 30000]  # Hz
SOURCE_ANGLES = [-35.0, 0.0, 40.0]   # degrees (initial)
N_SOURCES = len(SOURCE_ANGLES)

# ===============================================================
# 2. Geometry setup (Fermat spiral)
# ===============================================================
golden_angle = np.deg2rad(137.5)
aperture_radius = 0.025  # 5 cm radius (~10 cm diameter)
c_geom = aperture_radius / np.sqrt(N_MICS - 1)

x_coords, y_coords = [], []
for n in range(N_MICS):
    r = c_geom * np.sqrt(n)
    theta = n * golden_angle
    x_coords.append(r * np.cos(theta))
    y_coords.append(r * np.sin(theta))

x_coords = np.array(x_coords)
y_coords = np.array(y_coords)

# "Pitch" for ESPRIT (approx. radial spacing; crude but OK for test)
pitch = np.mean(np.diff(sorted(np.unique(np.sqrt(x_coords**2 + y_coords**2)))))

# Frequency axis (for picking closest bin)
f_axis = np.fft.rfftfreq(SAMPLES_PER_CHANNEL, 1 / SAMPLE_RATE_HZ)

# ===============================================================
# 3. Multi-source STM32 Frame Generator (deterministic)
# ===============================================================
def generate_fft_frame_from_dataframe(angle_degs: List[float]) -> FFTFrame:
    """
    Simulate one STM32 FFT frame with multiple plane-wave sources.

    angle_degs: list of DOAs (degrees) for each source; len == N_SOURCES
    """
    frame = FFTFrame()
    frame.channel_count = N_MICS
    frame.sampling_rate = SAMPLE_RATE_HZ
    frame.fft_size = SAMPLES_PER_CHANNEL
    frame.frame_id += 1

    t = np.arange(SAMPLES_PER_CHANNEL) / SAMPLE_RATE_HZ
    mic_signals = np.zeros((N_MICS, len(t)), dtype=np.float32)

    # Combine all sources
    for src_idx, angle_deg in enumerate(angle_degs):
        angle_rad = np.deg2rad(angle_deg)
        f = SOURCE_FREQS[src_idx]
        for i in range(N_MICS):
            delay = -(x_coords[i] * np.cos(angle_rad) +
                      y_coords[i] * np.sin(angle_rad)) / SPEED_SOUND
            delayed_t = t - delay
            mic_signals[i, :] += np.sin(2 * np.pi * f * delayed_t)

    # Add noise once
    mic_signals += np.random.normal(0, np.sqrt(NOISE_POWER), mic_signals.shape)

    # Convert to FFT domain (per-mic)
    fft_data = np.fft.rfft(mic_signals, axis=1)
    frame.fft_data = fft_data.astype(np.complex64)
    return frame

# ===============================================================
# 4. Beamforming algorithms
# ===============================================================
def music_spectrum(R: np.ndarray,
                   angles: np.ndarray,
                   f_signal: float,
                   n_sources: int) -> np.ndarray:
    """MUSIC spectrum vs. angle for a given covariance matrix."""
    eigvals, eigvecs = eigh(R)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    En = eigvecs[:, n_sources:]  # noise subspace

    spectrum = []
    for ang in angles:
        theta = np.deg2rad(ang)
        a = np.exp(
            -1j * 2 * np.pi * f_signal / SPEED_SOUND *
            -(x_coords * np.cos(theta) + y_coords * np.sin(theta))
        )
        a = a[:, np.newaxis]
        P = 1.0 / np.real(a.conj().T @ En @ En.conj().T @ a)
        spectrum.append(P[0, 0])

    spec = np.array(spectrum)
    spec /= np.max(spec) + 1e-12
    return spec


def esprit_estimate(R: np.ndarray,
                    f_signal: float,
                    n_sources: int) -> np.ndarray:
    """ESPIRIT DOA estimate (not used for heatmap, but handy to keep)."""
    eigvals, eigvecs = eigh(R)
    idx = eigvals.argsort()[::-1]
    Es = eigvecs[:, :n_sources]
    Es1, Es2 = Es[:-1], Es[1:]
    phi = pinv(Es1) @ Es2
    eigs_phi, _ = eig(phi)
    psi = np.angle(eigs_phi)
    val = -(psi * SPEED_SOUND) / (2 * np.pi * f_signal * pitch)
    val = np.clip(np.real(val), -1.0, 1.0)
    theta = np.arcsin(val)
    return np.degrees(theta)

# ===============================================================
# 5. Heatmap mapping
# ===============================================================
def spectra_to_heatmap(spec_matrix: np.ndarray,
                       out_width: int,
                       out_height: int) -> np.ndarray:
    """
    Map a [N_SOURCES x N_ANGLES] matrix into a 2D uint8 heatmap,
    then resize to (out_height, out_width).
    """
    # Normalize across entire matrix
    spec_min = float(spec_matrix.min())
    spec_max = float(spec_matrix.max())
    denom = spec_max - spec_min + 1e-12
    norm = (spec_matrix - spec_min) / denom  # 0..1

    heatmap_u8 = (255.0 * norm).astype(np.uint8)  # (N_SOURCES, N_ANGLES)

    # Resize to desired resolution (cv2 expects (width, height))
    resized = cv2.resize(heatmap_u8, (out_width, out_height),
                         interpolation=cv2.INTER_CUBIC)
    return resized

# ===============================================================
# 6. Main loop
# ===============================================================
def main():
    print("Acoustic Imager - Fermat Spiral Multi-Source Heatmap Test")
    print("=" * 70)

    # Background and FFmpeg
    background = create_background_frame(WIDTH, HEIGHT)

    ffmpeg_process = None
    if USE_FFMPEG:
        print("Setting up FFmpeg process...")
        ffmpeg_process = setup_ffmpeg_process(WIDTH, HEIGHT, FPS, OUTPUT_FILE)

    cv2.namedWindow("Fermat Heatmap (MUSIC, Multi-Source)", cv2.WINDOW_AUTOSIZE)

    frame_count = 0
    start_time = time.time()

    try:
        while frame_count < NUM_FRAMES:
            # Animate sources (slow sweep across FoV)
            for k in range(N_SOURCES):
                SOURCE_ANGLES[k] += (0.15 + 0.05 * k)
                if SOURCE_ANGLES[k] > 90.0:
                    SOURCE_ANGLES[k] = -90.0

            frame = generate_fft_frame_from_dataframe(SOURCE_ANGLES)

            # Build [N_SOURCES x N_ANGLES] MUSIC matrix
            spec_matrix = np.zeros((N_SOURCES, len(ANGLES)), dtype=np.float32)

            for k, f_sig in enumerate(SOURCE_FREQS):
                f_idx = int(np.argmin(np.abs(f_axis - f_sig)))
                Xf = frame.fft_data[:, f_idx][:, np.newaxis]  # (N_MICS, 1)
                R = Xf @ Xf.conj().T                           # (N_MICS, N_MICS)

                spec = music_spectrum(R, ANGLES, f_sig, n_sources=N_SOURCES)
                spec_matrix[k, :] = spec

            # Map spectra to heatmap image
            heatmap_2d = spectra_to_heatmap(spec_matrix, WIDTH, HEIGHT)

            # Overlay heatmap on background
            output_frame = apply_heatmap_overlay(heatmap_2d, background, ALPHA)

            # Draw some text overlay
            elapsed = time.time() - start_time
            cv2.putText(
                output_frame,
                f"Frame: {frame_count + 1}/{NUM_FRAMES}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                output_frame,
                f"t = {elapsed:.2f}s",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            angle_str = " | ".join(
                f"{f/1000:.1f} kHz: {ang:.1f}°"
                for f, ang in zip(SOURCE_FREQS, SOURCE_ANGLES)
            )
            cv2.putText(
                output_frame,
                angle_str,
                (10, HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # Show in OpenCV
            cv2.imshow("Fermat Heatmap (MUSIC, Multi-Source)", output_frame)

            # Send to FFmpeg if enabled
            if ffmpeg_process is not None:
                send_to_ffmpeg(output_frame, ffmpeg_process)

            # Handle key input / frame pacing
            key = cv2.waitKey(int(1000 // FPS)) & 0xFF
            if key == ord("q"):
                print("Quit requested by user.")
                break

            frame_count += 1

        total_time = time.time() - start_time
        print(f"Completed {frame_count} frames in {total_time:.2f} s")
        if frame_count > 0:
            print(f"Average FPS (sim): {frame_count / total_time:.2f}")

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        print("Cleaning up...")
        cv2.destroyAllWindows()
        if ffmpeg_process is not None:
            cleanup_ffmpeg_process(ffmpeg_process)
            print(f"Output video saved as: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
