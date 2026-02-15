# Acoustic Imager

Real-time acoustic beamforming visualization system with interactive controls.

## Quick Start

```bash
cd /path/to/Capstone_490_Software/src/software
python3 -m acoustic_imager.main
```

## Features

- **Real-time Beamforming**: MUSIC algorithm for direction-of-arrival estimation
- **Dual Data Sources**: 
  - SIM mode: Synthetic multi-source signals for testing
  - SPI mode: Hardware interface for real acoustic data
- **Interactive Bandpass Filter**: Drag frequency range handles in real-time
- **Camera Overlay**: Optional camera background (Picamera2, GStreamer, V4L2)
- **Video Recording**: Record visualization to MP4 with pause/resume
- **Performance Monitoring**: Built-in profiler for optimization
- **Configurable FPS**: 30 / 60 / MAX modes

## Architecture

The codebase is organized into focused modules:

```
acoustic_imager/
├── main.py              # Application entry point
├── config.py            # Configuration constants
├── state.py             # Runtime state
├── custom_types.py      # Shared data types
├── sources/             # Data source implementations
├── io/                  # Camera and SPI managers
├── dsp/                 # Signal processing (beamforming, heatmap)
├── spi/                 # SPI protocol layer
└── ui/                  # User interface components
```

See [REFACTORING_SUMMARY.md](./REFACTORING_SUMMARY.md) for detailed architecture documentation.

## Controls

### Mouse
- **Left Click**: Interact with buttons and menu
- **Drag**: Adjust bandpass filter range on frequency bar

### Keyboard
- **Q** or **ESC**: Quit application

### UI Buttons

**Bottom Bar:**
- **Camera**: Toggle camera overlay (ON/OFF/N/A)
- **Source**: Switch between SIM and SPI modes
- **DEBUG**: Toggle debug information overlay

**Top-Right Menu:**
- **30FPS / 60FPS / MAX**: Frame rate control
- **GAIN**: Gain mode (LOW/HIGH) - placeholder
- **SHOT**: Take screenshot
- **REC**: Start/stop recording
- **PAUSE**: Pause/resume recording

## Configuration

Edit `config.py` to customize:

```python
# Display
WIDTH = 1024
HEIGHT = 600

# Microphone Array
N_MICS = 16
SAMPLE_RATE_HZ = 100000

# Beamforming
SPEED_SOUND = 343.0  # m/s

# Visualization
REL_DB_MIN = -60.0
REL_DB_MAX = 0.0
```

## Dependencies

### Required
- Python 3.8+
- NumPy
- OpenCV (cv2)

### Optional
- `spidev`: For SPI hardware interface
- `picamera2`: For Raspberry Pi camera
- `ffmpeg`: For video recording

Install dependencies:
```bash
pip install numpy opencv-python
pip install spidev picamera2  # Optional
```

## Development

### Running Tests
```bash
cd src/software
python3 -c "from acoustic_imager import main; print('✅ All imports OK')"
```

### Module Structure

Each module is self-contained and can be imported independently:

```python
# DSP algorithms
from acoustic_imager.dsp.beamforming import music_spectrum
from acoustic_imager.dsp.heatmap import spectra_to_heatmap_absolute

# Data sources
from acoustic_imager.sources.sim_source import SimSource
from acoustic_imager.sources.spi_source import SPISource

# I/O managers
from acoustic_imager.io.camera_manager import CameraManager
```

### Adding a New Data Source

1. Create a new file in `sources/`
2. Implement the source interface:
   ```python
   class MySource:
       def read_frame(self) -> LatestFrame:
           # Return LatestFrame with fft_data
           pass
   ```
3. Import and use in `main.py`

### Adding a New Beamforming Algorithm

1. Add function to `dsp/beamforming.py`:
   ```python
   def my_algorithm(R, angles, f_signal, n_sources, ...) -> np.ndarray:
       # Return spectrum array
       pass
   ```
2. Call from main loop in `main.py`

## Performance

Typical performance on Raspberry Pi 4:
- **SIM mode**: 60 FPS sustained
- **SPI mode**: 30-60 FPS (depends on data rate)
- **With camera**: 30-45 FPS

Optimization tips:
- Use MAX FPS mode only when needed
- Disable camera if not required
- Reduce N_MICS for faster processing

## Troubleshooting

### Import Errors
```bash
# Add to PYTHONPATH
export PYTHONPATH="/path/to/Capstone_490_Software/src/software:$PYTHONPATH"
```

### Camera Not Detected
- Check permissions: `sudo usermod -a -G video $USER`
- Try different backends in `config.py`
- Verify camera with: `libcamera-hello` or `v4l2-ctl --list-devices`

### SPI Not Working
- Check device exists: `ls -l /dev/spidev*`
- Add user to spi group: `sudo usermod -a -G spi $USER`
- Enable SPI: `sudo raspi-config` → Interface Options → SPI

### Low FPS
- Check profiler output (printed every 60 frames)
- Identify bottleneck stages
- Consider reducing resolution or N_MICS

## License

[Your License Here]

## Contributors

[Your Team/Names Here]

## References

- MUSIC Algorithm: Schmidt, R. (1986). "Multiple emitter location and signal parameter estimation"
- ESPRIT Algorithm: Roy, R., & Kailath, T. (1989). "ESPRIT-estimation of signal parameters via rotational invariance techniques"

## Support

For issues, questions, or contributions, please contact [your contact info].

---

**Version**: 2.0 (Refactored)  
**Last Updated**: February 15, 2026
