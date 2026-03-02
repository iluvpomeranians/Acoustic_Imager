# config.py
"""
Central configuration/constants for the Acoustic Imager software.

This file mirrors the constants from the original monolithic script
so that splitting into modules does not change behavior.
"""

from __future__ import annotations

import struct
import numpy as np

# ===============================================================
# 1. Configuration
# ===============================================================
WINDOW_NAME = "Fermat Heatmap (SIM + SPI LOOPBACK) + Bandpass (MUSIC)"
N_MICS = 16
SAMPLES_PER_CHANNEL = 512

SAMPLE_RATE_HZ = 100000
SPEED_SOUND = 343.0
NOISE_POWER = 0.0005

WIDTH = 1024
HEIGHT = 600
ALPHA = 0.7
BUTTON_ALPHA = 0.92  # menu + children: reduce transparency (more opaque)

BUTTON_HITPAD_PX = 14

# Shared UI blue (BGR) for menu, HUD
MENU_ACTIVE_BLUE = (130, 82, 32)       # slightly darker blue
MENU_ACTIVE_BLUE_LIGHT = (165, 110, 48)  # gradient end

# Freq bar / bandpass: lighter neon blue so it stands out on dark bar
FREQ_BAR_BLUE = (230, 170, 80)  # BGR slightly darker blue for sliding window
# Gallery filter/sort modal: goldish yellow for selected option (bandpass-style accent)
MODAL_ACTIVE_GOLD = (0, 200, 255)  # BGR goldish yellow

FREQ_BAR_WIDTH = 150
DB_BAR_WIDTH = 80

# FPS defaults (MENU controls 30/60/MAX at runtime)
FPS_TARGET = 60
FPS_MODE_TO_TARGET = {"30": 30, "60": 60}

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = WIDTH
CAMERA_HEIGHT = HEIGHT
USE_CAMERA = True
USE_LIBCAMERA = True

# Camera post-processing
CAMERA_BRIGHTNESS_ALPHA = 1.20   # contrast multiplier (1.0 = no change)
CAMERA_BRIGHTNESS_BETA  = 15      # brightness offset (-100 to +100 typical)
DETECTION_BOX_COLOR = (0, 215, 255)
#DETECTION_BOX_COLOR = (150, 180, 0) #teal
#DETECTION_BOX_COLOR = (0, 200, 200) # yellow

# Bandpass
F_DISPLAY_MAX = 45000.0
F_MIN_HZ_DEFAULT = 24000.0
F_MAX_HZ_DEFAULT = 35000.0

DRAG_MARGIN_PX = 18

# Angle grid
ANGLES = np.linspace(-90, 90, 181)

# ===============================================================
# FFT frequency axis
# ===============================================================
f_axis = np.fft.rfftfreq(SAMPLES_PER_CHANNEL, 1 / SAMPLE_RATE_HZ)
N_BINS = SAMPLES_PER_CHANNEL // 2 + 1  # 257

# ===============================================================
# SIM sources
# ===============================================================
SIM_SOURCE_FREQS = [9000, 11000, 30000]
SIM_SOURCE_ANGLES = [-35.0, 0.0, 40.0]
SIM_SOURCE_AMPLS = [0.6, 1.0, 2.0]
SIM_N_SOURCES = len(SIM_SOURCE_ANGLES)

# dB mapping for heatmap intensity
REL_DB_MIN = -60.0
REL_DB_MAX = 0.0

# ===============================================================
# Blend acceleration (LUT + integer math)
# ===============================================================
# weight = uint8 0..255 derived from heatmap intensity (0..255)
# We fold in sqrt-gamma and ALPHA here so per-frame work is tiny.
BLEND_GAMMA = 0.5  # same as np.power(mask, 0.5)

_w_lut = np.arange(256, dtype=np.float32) / 255.0
_w_lut = np.power(_w_lut, BLEND_GAMMA) * float(ALPHA)  # 0..ALPHA
W_LUT_U8 = np.clip(np.round(_w_lut * 255.0), 0, 255).astype(np.uint8)

# optional: reuse these buffers to reduce allocs (left here for compatibility)
_tmp_w = None
_tmp_out = None

# ===============================================================
# 2. SPI frame format (float32 MAG + float32 PHASE)
# ===============================================================
MAGIC_START = 0x46524654  # 'TFRF' (example)
MAGIC_END = 0x454E4421    # 'END!' (example)
VERSION = 1

HEADER_FMT = "<IHH I HH I HH I"
HEADER_LEN = struct.calcsize(HEADER_FMT)

TRAILER_FMT = "<II"  # crc32(u32), end_magic(u32)
TRAILER_LEN = struct.calcsize(TRAILER_FMT)

PAYLOAD_LEN = N_MICS * N_BINS * 2 * 4  # 8 bytes per bin per mic
FRAME_BYTES = HEADER_LEN + PAYLOAD_LEN + TRAILER_LEN

# SPI config
SOURCE_DEFAULT = "SIM"   # or "SIM" / "SPI_HW"
SOURCE_MODES = ("SIM", "LOOP", "HW")

SPI_BUS = 0
SPI_DEV = 0
SPI_MODE = 0
SPI_BITS = 8

SPI_MAX_SPEED_HZ = 80_000_000
SPI_XFER_CHUNK = 8192

# For debugging if FPS is too slow, we can pre-generate a static frame and reuse it.
STATIC_TX_FRAME = None

# --- LOOPBACK SPI "virtual sources" ---
SPI_SIM_BINS = [35, 80, 160, 220]
SPI_SIM_AMPLS = [6.0, 3.0, 5.0, 4.0]  # boosted
SPI_SIM_ANGLES = [-25.0, 35.0, -5.0, 60.0]
SPI_SIM_DRIFT_DEG_PER_SEC = [1.2, -0.6, 0.4, -0.3]

CRC_EVERY_N = 30

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

pitch = (
    np.mean(np.diff(sorted(np.unique(np.sqrt(x_coords**2 + y_coords**2)))))
    if N_MICS > 2
    else 0.0
)