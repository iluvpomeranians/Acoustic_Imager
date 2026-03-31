# Acoustic Imager – Real-Time Sound Localization

## Overview

This repository contains an **acoustic imaging system** (COEN 490 capstone). A **16-microphone array** localizes sound sources and displays real-time heatmaps, with optional camera overlay, bandpass filtering, and multiple data sources (hardware SPI, simulated, loopback, reference).



https://github.com/user-attachments/assets/3cb0d738-56af-4772-b79c-725c6c1b056e


---

## Key Features

- **16-mic array**: Fermat spiral layout for simulation; measured geometry for hardware (HW/LOOP)
- **Real-time beamforming**: MUSIC and ESPRIT, with 2D MUSIC and covariance smoothing for HW
- **Data sources**: HW (SPI), SIM (simulated), LOOP (SPI loopback), REF (0 dB baseline)
- **Interactive heatmap**: Camera overlay, bandpass filter (draggable range), dB colorbar, colormap options (MAGMA, JET, TURBO, INFERNO)
- **Crosshairs**: Tap heatmap for crosshair with level/trend tooltip; optional 3 s / 12 s tracking
- **Spectrum analyzer**: Frequency bar with cursor; dB / NORM / dBA modes
- **UI**: Frame rate (30 / 60 / MAX), gain (LOW/HIGH), Wi‑Fi modal, settings modal, gallery, video recording, screenshots
- **Position & radar**: Optional compass (magnetometer), GPS, Wi‑Fi geolocation, radar map widget, directional history logging
- **Calibration**: Calibration suite (in-app), camera–array alignment constants, per-mic gain and HW geometry in config

---

## Project Objectives

- Achieve **~9° angular resolution** with a 16-mic array
- Manage spatial aliasing above ~8.6 kHz using **subspace methods** (MUSIC/ESPRIT)
- **Low-level acquisition** on STM32 with SPI transfer to Raspberry Pi
- **Real-time visualization** with interactive controls and multiple sources
- Deliver a **maintainable prototype** with modular software

---

## Team

- **David** – DSP lead, MUSIC/ESPRIT, visualization
- **Basem** – C++ DSP, software architecture
- **Fabian** – Requirements, integration testing
- **Tim** – Literature review, peripheral testing
- **Rob** – Hardware selection, PCB design
- **Ahmad** – Housing, assembly, CAD

---

## Quick Start

Run the application using the deploy script from the repository root:

```bash
sudo ./deploy/deploy_imager.sh
```

First run installs system packages (Python, OpenCV, libcamera, etc.), sets up the systemd service, and starts the UI. On later runs the script only toggles DEV mode or service state unless you request a clean reinstall.

**Clean reinstall** (remove existing service and reinstall):

```bash
sudo ./deploy/deploy_imager.sh --clean
```

Other options: `-status` to show service status, `-dev` to enable/disable DEV mode (skip auto-start for development).

---

## Calibration & Array Geometry

### Array geometry

- **Simulation (SIM)**: Fermat spiral in `config.py` — golden angle ~137.5°, aperture radius 1 cm, `x_coords` / `y_coords` in meters.
- **Hardware (HW / LOOP)**: Measured positions in **payload order** (0=U3 … 15=U14) as `x_coords_hw` and `y_coords_hw` in `config.py`. Generated from `utilities/calibration/array_geometry/`; see `array_geometry.md` for table, CSV, and MUSIC input reference.
- **Pitch**: Mean radial spacing (e.g. `pitch_hw` for HW) is used by the pipeline; geometry scripts output it for reference.

### Camera–array calibration

- In `config.py`: `CALIBRATION_DISTANCE_INCHES` (camera to array center), `CALIBRATION_OFFSET_INCHES` (lateral offset; positive = array right of camera center). Used for overlay alignment and tests. See `utilities/calibration/calibration_test.md` for the test setup.

### Per-mic and pipeline tuning

- **Per-mic gain**: `SPI_MIC_GAIN` in `config.py` (length 16); use `utilities/calibration/metrics_debug.py --live --write-config` to tune.
- **Calibration suite**: In-app via **Settings → Calibration Suite**; can run calibration flow and export results. Standalone app: `utilities/calibration/calibration_standalone_app.py`.

---

## User Interface & Controls

### Mouse

- **Left click**: Buttons, menu items, pills, bandpass handles (drag to set frequency range on the frequency bar).
- **Heatmap**: Tap to show crosshair and level/trend tooltip; tap again near crosshair to dismiss.
- **Spectrum bar**: Double-tap toggles cursor; single tap places or moves the dot.
- **Swipe down** (from top or bottom): Hide top HUD or bottom HUD + menu.
- **Swipe up**: Show HUD/menu. When menu is open, one swipe down closes menu; second swipe hides bottom HUD.
- **Swipe left**: Show menu (horizontal).
- **Single tap** in content (with menu open): Close menu.

### Keyboard

- **Q** or **ESC**: Quit.

### Bottom bar (pills)

- **SHOT**: Screenshot (saved to output folder; gallery refreshes).
- **REC**: Start recording; when recording, same pill toggles **PAUSE**; when paused, **Resume** (yellow) and **Stop** (red) appear.
- **Gallery**: Open gallery (grid of captures/videos; swipe/scroll, select, delete, tags, archive folders).

### Menu (dropdown)

- **Menu** button opens dropdown:
  - **Wi‑Fi**: Scan, connect, disconnect; password screen and on-screen keyboard.
  - **Settings**: Camera on/off, crosshairs, colormap, debug overlay, radar UI, map style (dark/light), position services, record compass history, **Calibration Suite**, email settings, flash firmware. Scrollable panel.
  - **30FPS / MAX**: Frame rate limit (30 or MAX; 60 removed in current UI).
  - **GAIN**: Toggle LOW / HIGH (drives hardware when gain control enabled).
  - **SRC**: Cycle source — HW → SIM → LOOP → REF.
  - **SPECTRUM**: Cycle spectrum mode — dB → NORM → dBA.

### Other UI

- **Top HUD**: Time, battery, FPS, network (e.g. SPI rate); tap pills to expand details.
- **Debug overlay**: Frame count, FPS, source stats, throughput (when debug enabled in settings).
- **Radar widget**: Optional mini map with heading and detections (when radar UI and position services enabled).

---

## Performance

- **FPS modes**: 30, 60 (if re-enabled), or MAX (no cap). Throttling applied each frame when not MAX.
- **Typical (Raspberry Pi 4)**: SIM can reach 60 FPS; HW/LOOP depends on SPI rate and MUSIC/heatmap cost (often 30–60 FPS). Camera overlay and 2D MUSIC reduce FPS.
- **Profiler**: Every 60 frames the main loop prints average stage timings (e.g. `read_source`, `heat_music`, `heat_draw`, `background`, `blend`, `bars`, `ui`, `imshow`, `waitKey`, `total` in ms). Use to find bottlenecks.
- **Tips**: Use 30 FPS when full rate isn’t needed; disable camera or radar if not used; tune `SPI_MUSIC_EVERY_N_FRAMES` and 2D coarse/fine resolution in `config.py` to trade smoothness for speed.

---

**Last updated**: March 2026
