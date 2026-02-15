# Acoustic Imager - Refactoring Summary

## Overview

The acoustic imager codebase has been successfully refactored from a monolithic ~1760-line `main.py` into a well-organized modular structure. This document summarizes the changes and provides guidance on the new architecture.

## Directory Structure

```
acoustic_imager/
├── main.py                 # Main application entry point (~450 lines)
├── config.py              # Configuration constants and parameters
├── state.py               # Runtime state management
├── custom_types.py        # Shared data types (LatestFrame, SourceStats)
│
├── sources/               # Data source implementations
│   ├── __init__.py
│   ├── sim_source.py     # Simulated signal source
│   └── spi_source.py     # SPI hardware source
│
├── io/                    # Input/Output managers
│   ├── __init__.py
│   ├── camera_manager.py # Camera capture (Picamera2, OpenCV)
│   └── spi_manager.py    # Low-level SPI communication
│
├── spi/                   # SPI protocol layer
│   ├── __init__.py
│   └── spi_protocol.py   # Frame format, validation, CRC
│
├── dsp/                   # Digital Signal Processing
│   ├── __init__.py
│   ├── beamforming.py    # MUSIC and ESPRIT algorithms
│   ├── heatmap.py        # Heatmap generation and blending
│   └── bars.py           # Frequency bar and dB colorbar
│
└── ui/                    # User Interface components
    ├── __init__.py
    ├── ui_components.py  # Buttons, menus, mouse handlers
    └── video_recorder.py # Video recording via ffmpeg
```

## Module Responsibilities

### Core Modules

#### `main.py` (Entry Point)
- **Purpose**: Application orchestration and main loop
- **Responsibilities**:
  - Initialize all subsystems (sources, camera, UI)
  - Main rendering loop
  - Mouse event handling
  - FPS throttling and profiling
- **Key Functions**:
  - `main()`: Main application loop
  - `mouse_callback()`: Handle mouse events for UI and bandpass dragging

#### `config.py` (Configuration)
- **Purpose**: Central configuration constants
- **Contents**:
  - Display settings (WIDTH, HEIGHT, ALPHA)
  - Microphone array geometry (N_MICS, x_coords, y_coords)
  - Signal processing parameters (SAMPLE_RATE_HZ, SPEED_SOUND)
  - SPI framing constants (MAGIC_START, MAGIC_END, HEADER_FMT)
  - UI layout constants (FREQ_BAR_WIDTH, DB_BAR_WIDTH)

#### `state.py` (Runtime State)
- **Purpose**: Mutable application state
- **Contents**:
  - `button_state`: UI state (recording, camera, source mode, FPS, gain)
  - Bandpass filter state (F_MIN_HZ, F_MAX_HZ, DRAG_ACTIVE)
  - Shared references (CURRENT_FRAME, OUTPUT_DIR, CAMERA_AVAILABLE)

#### `custom_types.py` (Data Types)
- **Purpose**: Shared data structures
- **Types**:
  - `SourceStats`: Source statistics (frames_ok, bad_parse, bad_crc, etc.)
  - `LatestFrame`: FFT frame container (fft_data, frame_id, ok, stats)

### Source Modules (`sources/`)

#### `sim_source.py` (Simulated Source)
- **Class**: `SimSource`
- **Purpose**: Generate synthetic multi-source signals for testing
- **Key Method**: `read_frame() -> LatestFrame`
- **Features**:
  - Configurable source frequencies, angles, amplitudes
  - Realistic noise injection
  - Deterministic angle drift for visualization

#### `spi_source.py` (SPI Source)
- **Class**: `SPISource`
- **Purpose**: High-level SPI source wrapper
- **Key Methods**:
  - `start()`: Start SPI acquisition thread
  - `stop()`: Stop SPI acquisition
  - `get_latest() -> LatestFrame`: Non-blocking latest frame retrieval
- **Features**: Wraps SPIManager for threaded operation

### I/O Modules (`io/`)

#### `camera_manager.py` (Camera Management)
- **Class**: `CameraManager`
- **Purpose**: Camera acquisition with automatic backend detection
- **Key Methods**:
  - `detect_and_open()`: Auto-detect camera (Picamera2, GStreamer, V4L2)
  - `start()`: Start background capture thread
  - `get_latest_frame()`: Non-blocking frame retrieval
- **Features**:
  - Threaded "latest frame" capture (main loop never blocks)
  - Multiple backend support
  - Automatic RGB/BGR conversion

#### `spi_manager.py` (SPI Manager)
- **Class**: `SPIManager`
- **Purpose**: Low-level SPI communication and framing
- **Key Methods**:
  - `start()`: Start SPI worker thread
  - `stop()`: Stop SPI worker
  - `get_latest() -> LatestFrame`: Thread-safe frame retrieval
- **Features**:
  - Frame validation (magic numbers, CRC)
  - Mag/phase to complex64 conversion
  - Threaded mailbox pattern

### SPI Protocol (`spi/`)

#### `spi_protocol.py` (Protocol Layer)
- **Purpose**: SPI frame format and validation
- **Features**:
  - Frame structure definition (header + payload + trailer)
  - CRC validation
  - Loopback frame generation (for testing)

### DSP Modules (`dsp/`)

#### `beamforming.py` (Beamforming Algorithms)
- **Functions**:
  - `music_spectrum()`: Vectorized MUSIC algorithm
  - `esprit_estimate()`: ESPRIT angle estimation
- **Features**:
  - No global state (all parameters passed explicitly)
  - Optimized for performance (vectorized operations)
  - Normalized output (max=1)

#### `heatmap.py` (Heatmap Generation)
- **Functions**:
  - `spectra_to_heatmap_absolute()`: Convert MUSIC spectra to heatmap
  - `build_w_lut_u8()`: Build blend weight lookup table
  - `blend_heatmap_left()`: Fast heatmap blending with background
- **Features**:
  - ROI-based Gaussian blob rendering (fast)
  - Integer-based blending (LUT optimization)
  - Configurable dB range mapping

#### `bars.py` (Visualization Bars)
- **Functions**:
  - `draw_frequency_bar()`: Frequency spectrum bar with bandpass indicators
  - `draw_db_colorbar()`: dB scale colorbar
  - `freq_to_y()`, `y_to_freq()`: Coordinate conversion helpers
- **Features**:
  - Interactive draggable handles
  - Frequency labels in kHz
  - Color-coded bandpass region

### UI Modules (`ui/`)

#### `ui_components.py` (UI Controls)
- **Classes**:
  - `Button`: Interactive button widget
  - `ButtonState`: UI state container (already in state.py, re-exported here)
- **Functions**:
  - `init_buttons()`, `init_menu_buttons()`: Layout initialization
  - `update_button_states()`: Hover state updates
  - `draw_buttons()`, `draw_menu()`: Rendering
  - `handle_button_click()`, `handle_menu_click()`: Event handling
  - `save_screenshot()`: Screenshot capture
- **Features**:
  - Bottom control bar (Camera, Source, Debug)
  - Top-right menu (FPS, Gain, Screenshot, Record, Pause)
  - Hover and active states
  - Callbacks for source switching

#### `video_recorder.py` (Video Recording)
- **Class**: `VideoRecorder`
- **Purpose**: Record visualization to MP4 via ffmpeg
- **Key Methods**:
  - `start_recording()`: Begin recording
  - `write_frame()`: Write frame to video
  - `pause_recording()`, `resume_recording()`: Pause/resume
  - `stop_recording()`: Finalize video
- **Features**:
  - H.264 encoding
  - Configurable FPS and quality
  - Frame counting

## Import Structure

All modules use absolute imports with the `acoustic_imager` package prefix:

```python
# Good (absolute imports)
from acoustic_imager import config
from acoustic_imager.sources.sim_source import SimSource
from acoustic_imager.dsp.beamforming import music_spectrum

# Bad (relative imports - DO NOT USE)
import config  # ❌
from sources.sim_source import SimSource  # ❌
```

## Key Improvements

### 1. **Modularity**
- Each module has a single, well-defined responsibility
- Easy to test individual components
- Clear separation of concerns

### 2. **Maintainability**
- ~1760 lines → ~450 lines in main.py + organized modules
- Easy to locate and modify specific functionality
- Self-documenting code structure

### 3. **Reusability**
- DSP algorithms can be used in other projects
- Camera and SPI managers are standalone
- UI components are decoupled from business logic

### 4. **Performance**
- No performance regression (same algorithms)
- Threaded I/O prevents blocking
- Optimized blending and beamforming preserved

### 5. **Extensibility**
- Easy to add new data sources (implement source interface)
- Easy to add new beamforming algorithms
- Easy to add new UI controls

## Migration from Old Code

If you have old code that imports from the monolithic `main.py`, here's how to migrate:

### Old (Monolithic)
```python
from main import (
    music_spectrum,
    spectra_to_heatmap_absolute,
    draw_frequency_bar,
    SimFFTSource,
    SpiFFTSource,
)
```

### New (Modular)
```python
from acoustic_imager.dsp.beamforming import music_spectrum
from acoustic_imager.dsp.heatmap import spectra_to_heatmap_absolute
from acoustic_imager.dsp.bars import draw_frequency_bar
from acoustic_imager.sources.sim_source import SimSource
from acoustic_imager.sources.spi_source import SPISource
```

## Running the Application

```bash
cd /path/to/Capstone_490_Software/src/software
python3 -m acoustic_imager.main
```

Or with the shebang:
```bash
cd /path/to/Capstone_490_Software/src/software/acoustic_imager
./main.py
```

## Testing Imports

To verify all imports work correctly:

```bash
cd /path/to/Capstone_490_Software/src/software
python3 -c "from acoustic_imager import main; print('✅ All imports OK')"
```

## Future Enhancements

Potential areas for further improvement:

1. **Configuration Management**
   - YAML/JSON config file support
   - Runtime configuration updates
   - Profile switching (dev/prod)

2. **Testing**
   - Unit tests for DSP algorithms
   - Integration tests for data sources
   - Mock camera/SPI for CI/CD

3. **Logging**
   - Replace print() with proper logging
   - Log levels (DEBUG, INFO, WARNING, ERROR)
   - Log file rotation

4. **Documentation**
   - API documentation (Sphinx)
   - User manual
   - Developer guide

5. **Plugin System**
   - Pluggable beamforming algorithms
   - Custom visualization modes
   - External data source plugins

## Troubleshooting

### Import Errors
If you see `ModuleNotFoundError: No module named 'acoustic_imager'`:
- Ensure you're running from the correct directory
- Add `src/software` to your PYTHONPATH:
  ```bash
  export PYTHONPATH="/path/to/Capstone_490_Software/src/software:$PYTHONPATH"
  ```

### Missing Dependencies
The code has fallbacks for optional external dependencies:
- `fftframe.py` → Minimal stub provided
- `heatmap_pipeline_test.py` → Gradient background fallback
- `stage_profiler.py` → No-op profiler stub

### Camera Not Detected
- Check `state.CAMERA_AVAILABLE` flag
- Try different backends (Picamera2, GStreamer, V4L2)
- Verify camera permissions

### SPI Not Working
- Check `SPIDEV_AVAILABLE` flag
- Verify SPI device permissions (`/dev/spidev0.0`)
- Ensure MOSI-MISO loopback for testing

## Contact & Support

For questions or issues with the refactored codebase, please refer to:
- Main README: `/src/software/README.md`
- Original documentation in the monolithic version (if preserved)

---

**Refactoring Date**: February 15, 2026  
**Original Code**: ~1760 lines (monolithic)  
**Refactored Code**: ~450 lines main + ~1500 lines in modules  
**Status**: ✅ Complete and tested
