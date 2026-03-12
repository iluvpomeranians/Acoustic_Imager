#!/usr/bin/env python3
"""
Array geometry for the Acoustic Imager (measured layout).

Structure matches config.py: Section 1 Configuration, Section 2 Raw data,
Section 3 Geometry setup. Exports N_MICS, SPEED_SOUND, x_coords, y_coords, pitch
so config can use this module as a drop-in geometry source (measured array
instead of Fermat spiral).

Source: FreeCAD measurements. CSV X,Y,Z = vector from camera to mic.
"Array Center → Camera" = vector from array center to camera. A 180° rotation
in the XY plane is applied so the view matches physical setup (camera top-left
of array). Run as script for table, plot, CSV, and MUSIC summary.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ===============================================================
# 1. Configuration
# ===============================================================
N_MICS = 16
SPEED_SOUND = 343.0
REF_FREQ_HZ = 30000.0  # reference for wavelength (middle of typical bandpass)

OUTPUT_CSV = Path(__file__).resolve().parent / "array_geometry.csv"

# ===============================================================
# 2. Raw measurement data (FreeCAD)
# ===============================================================
# Vector from array center to camera (mm)
ARRAY_CENTER_TO_CAMERA_MM = (47.05, 32.80, 8.64)
ARRAY_CENTER_TO_CAMERA_DIST_MM = 58.00

# Mics: (label, min_dist_mm, x_mm, y_mm, z_mm) in channel order; x,y,z = from camera to mic
# Channel order: 0=U1, 1=U2, 2=U3, 3=U4, 4=U6, 5=U7, 6=U8, 7=U9, 8=U11, 9=U12,
#                10=U13, 11=U14, 12=U16, 13=U17, 14=U18, 15=U19
MICS_RAW = [
    ("U1", 51.86, 45.00, 24.30, 8.64),
    ("U2", 37.61, 23.10, 28.40, 8.64),
    ("U3", 43.11, 33.60, 25.60, 8.64),
    ("U4", 40.99, 37.20, 14.90, 8.64),
    ("U6", 66.86, 58.50, 31.20, 8.64),
    ("U7", 50.59, 49.10, 8.60, 8.64),
    ("U8", 57.38, 53.70, 18.30, 8.64),
    ("U9", 69.41, 65.20, 22.20, 8.64),
    ("U11", 49.60, 28.60, 39.60, 8.64),
    ("U12", 80.43, 59.70, 53.20, 8.64),
    ("U13", 70.26, 49.90, 48.70, 8.64),
    ("U14", 67.24, 39.40, 53.80, 8.64),
    ("U16", 58.18, 39.00, 42.30, 8.64),
    ("U17", 57.34, 45.80, 33.40, 8.64),   # center most
    ("U18", 76.14, 63.70, 40.80, 8.64),
    ("U19", 63.75, 50.60, 37.80, 8.64),
]


def _to_array_center_xy(x_cam_mm: float, y_cam_mm: float) -> tuple[float, float]:
    """Convert camera-frame (from camera to point) to array-center frame XY (mm)."""
    cx, cy = ARRAY_CENTER_TO_CAMERA_MM[0], ARRAY_CENTER_TO_CAMERA_MM[1]
    return (x_cam_mm + cx, y_cam_mm + cy)


def _flip_180(x_mm: float, y_mm: float) -> tuple[float, float]:
    """180° rotation in XY plane so camera appears top-left in 2D view."""
    return (-x_mm, -y_mm)


# ===============================================================
# 3. Geometry setup (measured array, flipped frame)
# ===============================================================
# Camera in array-center frame then flipped
_cam_center_x = ARRAY_CENTER_TO_CAMERA_MM[0]
_cam_center_y = ARRAY_CENTER_TO_CAMERA_MM[1]
cam_fx, cam_fy = _flip_180(_cam_center_x, _cam_center_y)

x_mm_list: list[float] = []
y_mm_list: list[float] = []
geometry_table: list[dict] = []

for ch, (label, min_dist_mm, x_cam, y_cam, z_cam) in enumerate(MICS_RAW):
    x_center, y_center = _to_array_center_xy(x_cam, y_cam)
    x_f, y_f = _flip_180(x_center, y_center)

    x_mm_list.append(x_f)
    y_mm_list.append(y_f)

    dist_to_camera_mm = math.hypot(x_cam, y_cam, z_cam)
    dist_to_center_mm = math.hypot(x_center, y_center)
    azimuth_deg = math.degrees(math.atan2(y_f, x_f))
    x_m = x_f / 1000.0
    y_m = y_f / 1000.0

    geometry_table.append({
        "channel_index": ch,
        "mic_label": label,
        "x_mm": round(x_f, 2),
        "y_mm": round(y_f, 2),
        "x_m": x_m,
        "y_m": y_m,
        "dist_to_camera_mm": round(dist_to_camera_mm, 2),
        "dist_to_center_mm": round(dist_to_center_mm, 2),
        "azimuth_deg": round(azimuth_deg, 2),
        "min_dist_mm": min_dist_mm,
    })

# Arrays in meters (same names as config.py for drop-in use)
x_coords = np.array([r["x_m"] for r in geometry_table], dtype=np.float64)
y_coords = np.array([r["y_m"] for r in geometry_table], dtype=np.float64)

# Pitch: mean radial spacing (m), used by ESPRIT
radii = sorted(set(round(math.hypot(x, y), 6) for x, y in zip(x_mm_list, y_mm_list)))
if len(radii) > 1:
    pitch = (sum(radii[i + 1] - radii[i] for i in range(len(radii) - 1)) / (len(radii) - 1)) / 1000.0
else:
    pitch = 0.0

# Aperture and wavelength (for MUSIC sanity checks)
aperture_radius = max(math.hypot(x_mm, y_mm) for x_mm, y_mm in zip(x_mm_list, y_mm_list)) / 1000.0
wavelength_ref = SPEED_SOUND / REF_FREQ_HZ
half_wavelength = wavelength_ref / 2.0

pair_dists = []
for i in range(N_MICS):
    for j in range(i + 1, N_MICS):
        pair_dists.append(math.hypot(x_coords[i] - x_coords[j], y_coords[i] - y_coords[j]))
min_pair_m = min(pair_dists) if pair_dists else 0.0
max_pair_m = max(pair_dists) if pair_dists else 0.0
mean_pair_m = sum(pair_dists) / len(pair_dists) if pair_dists else 0.0


def run() -> None:
    """Print table, show 2D plot, write CSV, print MUSIC summary."""
    # ---- 2D plot (matplotlib popup) ----
    disp_cam_x, disp_cam_y = 0.0, 0.0
    disp_mic_x = [x_mm - cam_fx for x_mm in x_mm_list]
    disp_mic_y = [y_mm - cam_fy for y_mm in y_mm_list]
    min_dx = min(disp_cam_x, *disp_mic_x)
    min_dy = min(disp_cam_y, *disp_mic_y)
    margin_mm = 5.0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect("equal")
    ax.set_xlim(min_dx - margin_mm, 0 + margin_mm)
    ax.set_ylim(min_dy - margin_mm, 0 + margin_mm)
    ax.invert_xaxis()
    ax.set_xlabel("x (mm), camera at origin")
    ax.set_ylabel("y (mm)")
    ax.set_title("Array geometry — camera top-left, mic array (180° flip applied)")

    ax.plot(disp_cam_x, disp_cam_y, "k^", markersize=12, label="Camera", zorder=5)
    ax.annotate("Camera", (disp_cam_x, disp_cam_y), xytext=(5, -5), textcoords="offset points", fontsize=9)

    for (label, _, _, _, _), (dx, dy) in zip(MICS_RAW, zip(disp_mic_x, disp_mic_y)):
        ax.plot(dx, dy, "o", color="C0", markersize=6, zorder=3)
        ax.annotate(label, (dx, dy), xytext=(3, 3), textcoords="offset points", fontsize=8)

    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

    # ---- Table to stdout ----
    print("TABLE (flipped frame, array center = origin)")
    print("-" * 100)
    header = "ch  label   x_mm    y_mm    x_m      y_m      dist_cam_mm  dist_center_mm  azimuth_deg  min_dist_mm"
    print(header)
    print("-" * 100)
    for r in geometry_table:
        print(
            f"{r['channel_index']:2}   {r['mic_label']:<6} "
            f"{r['x_mm']:7.2f} {r['y_mm']:7.2f} "
            f"{r['x_m']:.6f} {r['y_m']:.6f} "
            f"{r['dist_to_camera_mm']:12.2f} {r['dist_to_center_mm']:15.2f} "
            f"{r['azimuth_deg']:11.2f} {r['min_dist_mm']:11.2f}"
        )
    print()

    # ---- Write CSV ----
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "channel_index", "mic_label", "x_mm", "y_mm", "x_m", "y_m",
                "dist_to_camera_mm", "dist_to_center_mm", "azimuth_deg", "min_dist_mm",
            ],
        )
        w.writeheader()
        w.writerows(geometry_table)
    print(f"Wrote table to: {OUTPUT_CSV}")
    print()

    # ---- MUSIC-ready summary ----
    print("MUSIC-ready summary (use in config/beamforming)")
    print("-" * 60)
    print("Channel order: 0=U1, 1=U2, 2=U3, 3=U4, 4=U6, 5=U7, 6=U8, 7=U9,")
    print("               8=U11, 9=U12, 10=U13, 11=U14, 12=U16, 13=U17, 14=U18, 15=U19")
    print()
    print("x_coords (m) = np.array([")
    print("    " + ", ".join(f"{x:.6f}" for x in x_coords))
    print("])")
    print()
    print("y_coords (m) = np.array([")
    print("    " + ", ".join(f"{y:.6f}" for y in y_coords))
    print("])")
    print()
    print(f"pitch (m) = {pitch:.6f}")
    print(f"speed_sound (m/s) = {SPEED_SOUND:.1f}")
    print(f"aperture_radius (m) = {aperture_radius:.6f}")
    print(f"wavelength at {REF_FREQ_HZ/1000:.0f} kHz (m) = {wavelength_ref:.6f}  (λ/2 = {half_wavelength:.6f})")
    print(f"pairwise distances (m): min = {min_pair_m:.6f}, max = {max_pair_m:.6f}, mean = {mean_pair_m:.6f}")
    print("  (keep max_pair < λ/2 to avoid grating lobes at ref frequency)")
    print()


if __name__ == "__main__":
    run()
