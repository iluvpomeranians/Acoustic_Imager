#!/usr/bin/env python3
"""
Acoustic Imager - Main Application

A real-time acoustic beamforming visualization system that supports:
- SIM (simulated) and SPI (hardware) data sources
- Interactive bandpass filtering
- MUSIC beamforming algorithm
- Camera background overlay
- Video recording and screenshots
- Configurable FPS and gain modes

This is the main entry point that orchestrates all modules.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import os

# ===============================================================
# Display Configuration for SSH/Headless Operation
# ===============================================================
# Uncomment ONE of these if running over SSH:
# os.environ['DISPLAY'] = ':0'              # Use local display on Pi (monitor attached)
# os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Headless mode (no display, recording only)

# ===============================================================
# Import modularized components
# ===============================================================
from acoustic_imager import config
from acoustic_imager import state
from acoustic_imager.custom_types import LatestFrame

# Data sources
from acoustic_imager.sources.sim_source import SimSource
from acoustic_imager.sources.spi_source import SPISource
from acoustic_imager.sources.spi_loopback_source import SPILoopbackSource
from acoustic_imager.io.spi_manager import SPIManager


# DSP modules
from acoustic_imager.dsp.beamforming import music_spectrum
from acoustic_imager.dsp.heatmap import (
    spectra_to_heatmap_absolute,
    build_w_lut_u8,
    blend_heatmap_left,
)
from acoustic_imager.dsp.bars import (
    draw_frequency_bar,
    draw_db_colorbar,
    freq_to_y,
    y_to_freq,
)

from acoustic_imager.ui.hud import draw_hud, handle_hud_click
from acoustic_imager.state import HUD

# I/O managers
from acoustic_imager.io.camera_manager import CameraManager

# UI components
from acoustic_imager.ui.ui_components import (
    button_state,
    buttons,
    menu_buttons,
    init_buttons,
    init_menu_buttons,
    update_button_states,
    draw_buttons,
    draw_menu,
    draw_recording_timestamp,
    handle_button_click,
    handle_gallery_click,
    draw_gallery_view,
    FPS_MODE_TO_TARGET,
)
from acoustic_imager.ui.video_recorder import VideoRecorder

# ===============================================================
# External dependencies (from parent directories)
# Optional - provide fallbacks if not available
# ===============================================================
try:
    sys.path.append(str(Path(__file__).resolve().parents[1] / "dataframe"))
    from fftframe import FFTFrame  # type: ignore
except ImportError:
    # Fallback: minimal FFTFrame stub
    class FFTFrame:
        def __init__(self):
            self.channel_count = 0
            self.sampling_rate = 0
            self.fft_size = 0
            self.frame_id = 0
            self.fft_data = None

try:
    from heatmap_pipeline_test import create_background_frame  # type: ignore
except ImportError:
    # Fallback: create simple gradient background
    def create_background_frame(width: int, height: int) -> np.ndarray:
        """Create a simple gradient background."""
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            val = int(20 + (y / height) * 30)
            bg[y, :] = (val, val, val)
        return bg

try:
    sys.path.append(str(Path(__file__).resolve().parents[3] / "utilities"))
    from stage_profiler import StageProfiler  # type: ignore
except ImportError:
    # Fallback: minimal profiler stub
    class StageProfiler:
        def __init__(self, keep=120):
            self.stages = {}

        def start_frame(self):
            pass

        def mark(self, stage: str):
            pass

        def end_frame(self):
            pass

        def ms(self, stage: str) -> float:
            return 0.0


# ===============================================================
# Mouse callback for interactive controls
# ===============================================================
def mouse_callback(event, x: int, y: int, flags, param) -> None:
    """
    Handle mouse events for:
    - Button clicks (camera, source, debug, menu, etc.)
    - Bandpass filter dragging (frequency bar handles)
    - Gallery view navigation
    """
    global video_recorder

    left_width, h = param
    mx, my = x, y

    # Update hover states
    state.CURSOR_POS = (mx, my)
    update_button_states(mx, my)

    # Handle left button down
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if we're in gallery grid view for drag scrolling
        if button_state.gallery_open and button_state.gallery_viewer_mode == "grid":
            # Check if click is not on back button or thumbnail
            back_button_clicked = "gallery_back" in menu_buttons and menu_buttons["gallery_back"].contains(mx, my)
            
            if not back_button_clicked:
                # Check if clicking on a thumbnail
                thumbnail_clicked = False
                if hasattr(button_state, 'gallery_thumbnail_rects'):
                    for thumb in button_state.gallery_thumbnail_rects:
                        if (thumb['x'] <= mx <= thumb['x'] + thumb['w'] and
                            thumb['y'] <= my <= thumb['y'] + thumb['h']):
                            thumbnail_clicked = True
                            break
                
                # If not clicking button or thumbnail, start drag
                if not thumbnail_clicked:
                    button_state.gallery_drag_active = True
                    button_state.gallery_drag_start_y = my
                    button_state.gallery_drag_start_offset = button_state.gallery_scroll_offset
                    return
        
        # Check gallery view first (if open)
        if handle_gallery_click(mx, my, state.OUTPUT_DIR):
            return

        # 1) HUD click handling (top priority, before buttons)
        try:
            from acoustic_imager.state import HUD
            from acoustic_imager.ui.hud import handle_hud_click
            # you need access to latest hud_rects -> simplest: store it in state each frame
            if event == cv2.EVENT_LBUTTONDOWN and hasattr(state, "HUD_RECTS"):
                new_panel = handle_hud_click(mx, my, state.HUD_RECTS, HUD.open_panel)
                if new_panel != HUD.open_panel:
                    HUD.open_panel = new_panel
                    return
        except Exception:
            pass
        # Check UI buttons first
        for b in buttons.values():
            if b.contains(mx, my):
                video_recorder = handle_button_click(
                    mx, my,
                    current_frame=state.CURRENT_FRAME,
                    output_dir=state.OUTPUT_DIR,
                    camera_available=state.CAMERA_AVAILABLE,
                    video_recorder=video_recorder,
                    width=config.WIDTH,
                    height=config.HEIGHT,
                )
                return

        # Check menu buttons
        if "menu" in menu_buttons and menu_buttons["menu"].contains(mx, my):
            video_recorder = handle_button_click(
                mx, my,
                current_frame=state.CURRENT_FRAME,
                output_dir=state.OUTPUT_DIR,
                camera_available=state.CAMERA_AVAILABLE,
                video_recorder=video_recorder,
                width=config.WIDTH,
                height=config.HEIGHT,
            )
            return

        if button_state.menu_open:
            for k, b in menu_buttons.items():
                if k == "menu":
                    continue
                if b.contains(mx, my):
                    video_recorder = handle_button_click(
                        mx, my,
                        current_frame=state.CURRENT_FRAME,
                        output_dir=state.OUTPUT_DIR,
                        camera_available=state.CAMERA_AVAILABLE,
                        video_recorder=video_recorder,
                        width=config.WIDTH,
                        height=config.HEIGHT,
                    )
                    return

        # Bandpass drag on frequency bar
        bar_left = left_width
        if mx >= bar_left and mx < config.WIDTH:
            y_min = freq_to_y(state.F_MIN_HZ, h, config.F_DISPLAY_MAX)
            y_max = freq_to_y(state.F_MAX_HZ, h, config.F_DISPLAY_MAX)
            dmin = abs(my - y_min)
            dmax = abs(my - y_max)

            state.DRAG_TARGET = "min" if dmin <= dmax else "max"
            state.DRAG_ACTIVE = True

            f = y_to_freq(my, h, config.F_DISPLAY_MAX)
            if state.DRAG_TARGET == "min":
                state.F_MIN_HZ = min(f, state.F_MAX_HZ)
            else:
                state.F_MAX_HZ = max(f, state.F_MIN_HZ)

    # Handle mouse move (dragging)
    elif event == cv2.EVENT_MOUSEMOVE:
        # Handle gallery drag scrolling
        if button_state.gallery_drag_active:
            drag_distance = my - button_state.gallery_drag_start_y
            # Drag down = positive distance = scroll up (decrease offset)
            # Drag up = negative distance = scroll down (increase offset)
            button_state.gallery_scroll_offset = button_state.gallery_drag_start_offset - drag_distance
            # Clamping is done in draw function
            return
        
        # Handle frequency bar dragging
        bar_left = left_width
        if state.DRAG_ACTIVE and mx >= bar_left and mx < config.WIDTH:
            f = y_to_freq(my, h, config.F_DISPLAY_MAX)
            if state.DRAG_TARGET == "min":
                state.F_MIN_HZ = min(f, state.F_MAX_HZ)
            elif state.DRAG_TARGET == "max":
                state.F_MAX_HZ = max(f, state.F_MIN_HZ)

    # Handle left button up
    elif event == cv2.EVENT_LBUTTONUP:
        # End gallery drag
        if button_state.gallery_drag_active:
            button_state.gallery_drag_active = False
            return
        
        # End frequency bar drag
        state.DRAG_ACTIVE = False
        state.DRAG_TARGET = None

    # Clamp frequency values
    state.F_MIN_HZ = float(np.clip(state.F_MIN_HZ, 0.0, config.F_DISPLAY_MAX))
    state.F_MAX_HZ = float(np.clip(state.F_MAX_HZ, 0.0, config.F_DISPLAY_MAX))
    if state.F_MIN_HZ > state.F_MAX_HZ:
        state.F_MIN_HZ, state.F_MAX_HZ = state.F_MAX_HZ, state.F_MIN_HZ


# ===============================================================
# Main application
# ===============================================================
def main() -> None:
    """Main application loop."""
    global video_recorder

    prev_mode = button_state.source_mode

    # ---- Profiler (for performance monitoring) ----
    prof = StageProfiler(keep=120)
    PRINT_EVERY = 60  # frames

    print("=" * 70)
    print("Acoustic Imager - Real-Time Beamforming Visualization")
    print("=" * 70)
    print(f"Resolution: {config.WIDTH}x{config.HEIGHT}")
    print(f"Mics: {config.N_MICS} | FFT bins: {config.N_BINS}")
    print(f"Sample rate: {config.SAMPLE_RATE_HZ} Hz")
    print(f"FPS modes: 30 / 60 / MAX (default: {button_state.fps_mode})")
    print("=" * 70)

    # ---- Setup output directory ----
    # Use data folder at repository root
    repo_root = Path(__file__).resolve().parents[3]  # Go up to Capstone_490_Software/
    state.OUTPUT_DIR = repo_root / "data" / "heatmap_captures"
    state.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {state.OUTPUT_DIR}")
    print()

    # ---- Initialize data sources ----
    global sim_source, spi_hw, spi_loopback

    sim_source = SimSource()

    spi_loopback = SPILoopbackSource()

    #TODO: DEMO ONLY USB
    from acoustic_imager.sources.usb_source import USBSource
    spi_hw = USBSource(port="/dev/ttyACM0", baud=115200)

    #TODO: FOR REAL SPI, USE THIS INSTEAD OF USBSource
    #spi_hw = SPISource(SPIManager(use_frame_ready=True))


    # Force initial mode from config if UI didn't set it yet
    if button_state.source_mode not in config.SOURCE_MODES:
        button_state.source_mode = config.SOURCE_DEFAULT

    prev_mode = button_state.source_mode

    # Start the selected SPI provider (if any)
    if button_state.source_mode == "SPI_LOOPBACK":
        spi_loopback.start()
    elif button_state.source_mode == "SPI_HW":
        spi_hw.start()

    # ---- Initialize camera ----
    camera_mgr = CameraManager()
    camera_mgr.detect_and_open()
    if state.CAMERA_AVAILABLE and button_state.camera_enabled:
        camera_mgr.start()

    detector = None
    if state.CAMERA_AVAILABLE and button_state.camera_enabled:
        #detector = DetectorClient(target_fps=8.0)

        def frame_provider():
            f = camera_mgr.get_latest_frame()
            if f is None or getattr(f, "size", 0) == 0:
                return None
            return f   # always BGR now

        #detector.start(frame_provider)

    # ---- Create static background ----
    background_full = create_background_frame(config.WIDTH, config.HEIGHT)
    left_width = config.WIDTH - config.FREQ_BAR_WIDTH

    # ---- Preallocated buffers (optimization) ----
    base_frame = np.empty((config.HEIGHT, config.WIDTH, 3), dtype=np.uint8)
    cam_bgr = np.empty((config.HEIGHT, config.WIDTH, 3), dtype=np.uint8)

    # ---- Build blend LUT ----
    w_lut_u8 = build_w_lut_u8(config.ALPHA, config.BLEND_GAMMA)

    # ---- Initialize video recorder ----
    global video_recorder
    video_recorder = VideoRecorder(state.OUTPUT_DIR, config.WIDTH, config.HEIGHT, fps=30)

    # ---- Setup OpenCV window ----
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    
    # Try to detect screen resolution using different methods
    try:
        # Method 1: Try to get screen info from environment
        import subprocess
        result = subprocess.run(['xrandr'], capture_output=True, text=True, timeout=1)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if '*' in line:  # Current resolution marked with *
                    parts = line.split()
                    for part in parts:
                        if 'x' in part and part[0].isdigit():
                            w, h = part.split('x')
                            actual_width = int(w)
                            actual_height = int(h.split('+')[0])
                            print(f"Detected screen resolution (xrandr): {actual_width}x{actual_height}")
                            config.WIDTH = actual_width
                            config.HEIGHT = actual_height
                            break
    except Exception as e:
        print(f"Could not detect screen resolution via xrandr: {e}")
    
    # Set fullscreen
    cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Recalculate dependent values with updated dimensions
    left_width = config.WIDTH - config.FREQ_BAR_WIDTH
    
    # Recreate buffers with correct size
    base_frame = np.empty((config.HEIGHT, config.WIDTH, 3), dtype=np.uint8)
    cam_bgr = np.empty((config.HEIGHT, config.WIDTH, 3), dtype=np.uint8)
    background_full = create_background_frame(config.WIDTH, config.HEIGHT)
    
    # Reinitialize UI with new dimensions
    init_buttons(left_width, state.CAMERA_AVAILABLE)
    init_menu_buttons(left_width)
    
    print(f"Final resolution: {config.WIDTH}x{config.HEIGHT}")
    
    cv2.setMouseCallback(config.WINDOW_NAME, mouse_callback, param=(left_width, config.HEIGHT))

    # ---- Initialize UI ----
    button_state.debug_enabled = False
    init_buttons(left_width, state.CAMERA_AVAILABLE)
    init_menu_buttons(left_width)

    # ---- Loop state ----
    frame_count = 0
    start_time = time.time()
    last_spi_fft_data: Optional[np.ndarray] = None

    # FPS tracking
    fps_ema = 0.0
    last_t = time.perf_counter()
    next_tick = time.perf_counter()

    try:
        while True:
            prof.start_frame()
            elapsed = time.time() - start_time

            # ---- FPS estimation ----
            now_t = time.perf_counter()
            dt = now_t - last_t
            last_t = now_t
            inst_fps = (1.0 / dt) if dt > 1e-6 else 0.0
            fps_ema = (0.92 * fps_ema + 0.08 * inst_fps) if fps_ema > 0 else inst_fps
            prof.mark("fps_est")

            # ---- FPS throttling ----
            if button_state.fps_mode in ("30", "60"):
                fps_target = FPS_MODE_TO_TARGET[button_state.fps_mode]
                period = 1.0 / max(1, fps_target)
                next_tick += period
                sleep = next_tick - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_tick = time.perf_counter()
            else:
                # MAX mode = unthrottled
                next_tick = time.perf_counter()
            prof.mark("throttle")

            mode = button_state.source_mode
            if mode != prev_mode:
                # stop both (safe)
                spi_loopback.stop()
                spi_hw.stop()

                last_spi_fft_data = None

                # start whichever is selected
                if mode == "SPI_LOOPBACK":
                    spi_loopback.start()
                elif mode == "SPI_HW":
                    spi_hw.start()

                prev_mode = mode

            # ---- Get current bandpass range ----
            f_min = state.F_MIN_HZ
            f_max = state.F_MAX_HZ

            # ---- Read FFT data from source ----
            mode = button_state.source_mode

            if mode == "SIM":
                latest_frame = sim_source.read_frame()
                source_label = "SIM"
            elif mode == "SPI_LOOPBACK":
                latest_frame = spi_loopback.get_latest()
                source_label = "SPI_LOOPBACK"
            else:  # "SPI_HW"
                latest_frame = spi_hw.get_latest()
                source_label = "SPI_HW"

            source_stats = latest_frame.stats
            fft_data = latest_frame.fft_data if latest_frame.ok else None

            # Fallback to last known data if current read failed
            if fft_data is None:
                if source_label.startswith("SPI") and last_spi_fft_data is not None:
                    fft_data = last_spi_fft_data
                else:
                    fft_data = np.zeros((config.N_MICS, config.N_BINS), dtype=np.complex64)
            elif source_label.startswith("SPI"):
                last_spi_fft_data = fft_data

            prof.mark("read_source")

            # ---- Beamforming + Heatmap generation ----
            if source_label == "SIM":
                # Filter sources by bandpass
                selected_indices = [
                    i for i, f in enumerate(config.SIM_SOURCE_FREQS)
                    if f_min <= f <= f_max
                ]

                if not selected_indices:
                    heatmap_left = np.zeros((config.HEIGHT, left_width), dtype=np.uint8)
                else:
                    n_sel = len(selected_indices)
                    spec_matrix = np.zeros((n_sel, len(config.ANGLES)), dtype=np.float32)
                    power = np.zeros(n_sel, dtype=np.float32)

                    running_max_power = 1e-12
                    for row_idx, src_idx in enumerate(selected_indices):
                        f_sig = float(config.SIM_SOURCE_FREQS[src_idx])
                        f_idx = int(np.argmin(np.abs(config.f_axis - f_sig)))
                        Xf = fft_data[:, f_idx][:, np.newaxis]
                        R = Xf @ Xf.conj().T

                        spec_matrix[row_idx, :] = music_spectrum(
                            R, config.ANGLES, f_sig, n_sel,
                            config.x_coords, config.y_coords, config.SPEED_SOUND
                        )
                        p = float(np.sum(np.abs(Xf) ** 2).real)
                        power[row_idx] = p
                        running_max_power = max(running_max_power, p)

                    power_rel = power / (running_max_power + 1e-12)
                    heatmap_left = spectra_to_heatmap_absolute(
                        spec_matrix, power_rel, left_width, config.HEIGHT,
                        config.REL_DB_MIN, config.REL_DB_MAX
                    )

            else:  # SPI mode
                # TODO: FOR SPI, TOP K BINS
                # FOR LOOPBACK ONLY:Filter bins by bandpass
                bins = [
                    b for b in config.SPI_SIM_BINS
                    if 0 <= b < config.N_BINS and (f_min <= float(config.f_axis[b]) <= f_max)
                ]

                if not bins:
                    heatmap_left = np.zeros((config.HEIGHT, left_width), dtype=np.uint8)
                else:
                    spec_matrix = np.zeros((len(bins), len(config.ANGLES)), dtype=np.float32)
                    power = np.zeros(len(bins), dtype=np.float32)

                    for i, f_idx in enumerate(bins):
                        f_sig = float(config.f_axis[f_idx])
                        Xf = fft_data[:, f_idx][:, np.newaxis]
                        R = Xf @ Xf.conj().T

                        spec_matrix[i, :] = music_spectrum(
                            R, config.ANGLES, f_sig, len(bins),
                            config.x_coords, config.y_coords, config.SPEED_SOUND
                        )
                        power[i] = float(np.sum(np.abs(Xf) ** 2).real)

                    # Brighter SPI: per-frame normalization
                    power_rel = power / (power.max() + 1e-12)
                    power_rel = np.power(power_rel, 0.6)
                    heatmap_left = spectra_to_heatmap_absolute(
                        spec_matrix, power_rel, left_width, config.HEIGHT,
                        config.REL_DB_MIN, config.REL_DB_MAX
                    )

            prof.mark("beamform+heatmap")

            # ---- Background (camera or static) ----
            if button_state.camera_enabled and state.CAMERA_AVAILABLE:
                cam_frame = camera_mgr.get_latest_frame()
                if cam_frame is not None and getattr(cam_frame, "size", 0) > 0:
                    try:
                        if camera_mgr.camera_type == "picamera2":
                            # already BGR
                            if cam_frame.shape[1] == config.WIDTH and cam_frame.shape[0] == config.HEIGHT:
                                base_frame[:] = cam_frame
                            else:
                                cv2.resize(cam_frame, (config.WIDTH, config.HEIGHT),
                                        dst=base_frame, interpolation=cv2.INTER_LINEAR)
                        else:
                            # opencv backend already BGR
                            if cam_frame.shape[:2] == (config.HEIGHT, config.WIDTH):
                                base_frame[:] = cam_frame
                            else:
                                cv2.resize(cam_frame, (config.WIDTH, config.HEIGHT), dst=base_frame, interpolation=cv2.INTER_LINEAR)
                    except Exception:
                        base_frame[:] = background_full
                else:
                    base_frame[:] = background_full
            else:
                base_frame[:] = background_full

            prof.mark("background")

            # ---- Blend heatmap onto background ----
            output_frame = blend_heatmap_left(base_frame, heatmap_left, left_width, w_lut_u8)
            prof.mark("blend")

            # ---- Draw frequency bar and dB colorbar ----
            draw_frequency_bar(
                output_frame, fft_data, config.f_axis, f_min, f_max,
                config.FREQ_BAR_WIDTH, config.F_DISPLAY_MAX
            )
            draw_db_colorbar(output_frame, config.REL_DB_MIN, config.REL_DB_MAX, config.DB_BAR_WIDTH)
            prof.mark("bars")

            # ---- Draw debug info ----
            if button_state.debug_enabled:

                text_x = config.DB_BAR_WIDTH + 12

                cv2.putText(output_frame, f"Frame: {frame_count}", (text_x, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(output_frame, f"t = {elapsed:.2f}s", (text_x, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(output_frame, f"Source: {source_label}", (text_x, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if source_label.startswith("SPI"):
                    mhz = (source_stats.sclk_hz_rep / 1e6) if source_stats.sclk_hz_rep else 0
                    y0 = 120
                    dy = 22

                    bytes_per_s = config.FRAME_BYTES * fps_ema
                    mbps_bytes = bytes_per_s / 1e6
                    mbps_bits = (bytes_per_s * 8) / 1e6

                    cv2.putText(output_frame, f"SPI {mhz:.0f}MHz", (text_x, y0),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    cv2.putText(output_frame, f"ok: {source_stats.frames_ok}", (text_x, y0 + 1*dy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    cv2.putText(output_frame, f"badParse: {source_stats.bad_parse}   badCRC: {source_stats.bad_crc}",
                               (text_x, y0 + 2*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    cv2.putText(output_frame, f"FPS: {fps_ema:5.1f}", (text_x, y0 + 3*dy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    cv2.putText(output_frame, f"Throughput: {mbps_bytes:.2f} MB/s  ({mbps_bits:.1f} Mb/s)",
                               (text_x, y0 + 4*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    if source_stats.last_err:
                        cv2.putText(output_frame, f"lastErr: {source_stats.last_err[:60]}",
                                   (text_x, y0 + 5*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # ---- Draw UI buttons ----
            draw_buttons(output_frame)
            draw_menu(output_frame)
            draw_recording_timestamp(output_frame, video_recorder)

            hud_rects = draw_hud(
                output_frame,
                details_level=HUD.details_level,
                open_panel=HUD.open_panel,
                fps_ema=fps_ema,
                elapsed_s=elapsed,
                frame_count=frame_count,
                source_label=source_label,
                source_stats=source_stats,
                fps_mode=button_state.fps_mode,
                frame_bytes=config.FRAME_BYTES,
            )

            state.HUD_RECTS = hud_rects

            # ---- Draw gallery view if open ----
            if button_state.gallery_open:
                draw_gallery_view(output_frame, state.OUTPUT_DIR)

            prof.mark("ui")

            # ---- Store current frame for screenshots ----
            state.CURRENT_FRAME = output_frame.copy()
            prof.mark("copy_frame")

            # ---- Record video if active ----
            if button_state.is_recording and video_recorder:
                video_recorder.write_frame(output_frame)
            prof.mark("record")

            # ---- Display frame ----
            cv2.imshow(config.WINDOW_NAME, output_frame)
            prof.mark("imshow")

            # ---- Handle keyboard input ----
            key = cv2.waitKey(1) & 0xFF
            prof.mark("waitKey")

            prof.end_frame()

            # ---- Print profiler stats periodically ----
            if (frame_count % PRINT_EVERY) == 0 and frame_count > 0:
                print(
                    "ms avg | "
                    f"read={prof.ms('read_source'):.2f} "
                    f"heat={prof.ms('beamform+heatmap'):.2f} "
                    f"bg={prof.ms('background'):.2f} "
                    f"blend={prof.ms('blend'):.2f} "
                    f"bars={prof.ms('bars'):.2f} "
                    f"ui={prof.ms('ui'):.2f} "
                    f"imshow={prof.ms('imshow'):.2f} "
                    f"waitKey={prof.ms('waitKey'):.2f} "
                    f"total={prof.ms('frame_total'):.2f}"
                )

            # ---- Check for exit ----
            if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("q") or key == 27:  # 'q' or ESC
                break

            frame_count += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("Cleaning up...")

        # Stop and cleanup resources
        if video_recorder:
            video_recorder.cleanup()

        spi_loopback.stop()
        spi_hw.stop()

        camera_mgr.close()

        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
