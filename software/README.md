# Software

This folder contains all application-level code for acquisition, preprocessing, DSP, and visualization.  
Each subfolder corresponds to a story area, and tasks are organized by Task ID from the backlog.

---

### Acquisition
- **SF-03-01: Raw Data Acquisition**  
  Write scripts to capture multi-channel audio streams sent from the STM32.  
  Data should be buffered in real-time, saved as `.wav` for debugging, and converted into NumPy arrays for the DSP pipeline.  
  Deliverable: `capture.py` and validation plots showing aligned channels.

- **EF-04-01: Data Transmission (Client Side)**  
  Implement Pi-side receiver scripts to read the data packets from STM32 (I²S/SPI/UART).  
  Verify no data loss occurs by checking sequence counters or headers.

---

### Preprocessing
- **SF-03-02: Preprocessing & Filtering (100 Hz – 20 kHz)**  
  Apply band-pass filtering and normalization to incoming signals.  
  This prepares data for FFT and beamforming by removing DC offset and unwanted noise.  
  Deliverable: Filtering module that outputs clean signals.

---

### DSP
- **SF-03-03: FFT Implementation**  
  Implement efficient FFT routines for each audio frame.  
  Validate with synthetic signals (e.g., sine waves) before moving to live data.

- **SF-03-04: Beamforming (Delay-and-Sum)**  
  Develop baseline beamforming using mic array geometry.  
  Test by localizing simple tone sources at known angles.

- **SF-03-05: Advanced DSP Algorithms (MUSIC/ESPRIT)**  
  Prototype higher-resolution localization algorithms for >9° accuracy.  
  Requires covariance matrix estimation and eigenvalue decomposition.

- **SF-03-06: Performance Benchmarking**  
  Profile equivalent pipelines in Python vs C++ (FFT, filtering, beamforming).  
  Measure throughput, latency, and jitter.  
  Deliverable: Benchmarks showing which implementation meets real-time constraints.

---

### Visualization
- **SF-04-01: Heatmap**  
  Convert processed signals into a 2D heatmap and overlay with video frames.  
  Use OpenCV for blending and color mapping.

- **SF-04-02: Frame Synchronization**  
  Ensure audio and video streams are aligned using timestamps.  
  Implement a ring buffer and validate sync with a clap/LED test.

- **SF-04-03: UI Development**  
  Add device controls for start/stop, recording, and playback.  
  Ensure responsiveness and simple interface.

- **SF-04-04: Visualization Latency Testing**  
  Measure update rate of the heatmap overlay.  
  Confirm ≥10 FPS and end-to-end latency <100 ms.
