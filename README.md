edit
---

## 🔬 Scientific Basis
### Array Geometry
- **Layout**: 16 microphones in a **Fermat’s spiral** (golden angle ~137.5°).
- **Why Fermat?**
  - Isotropy (uniform angular resolution in all directions).
  - Irregular spacing smears aliasing into noise floor → MUSIC/ESPRIT converge better.
  - Inspired by **phyllotaxis in nature** (sunflower seed packing).

### Key Equations
- Angular resolution:
  \[
  \Delta \theta \approx \frac{\lambda}{D}
  \]
- Spatial aliasing cutoff:
  \[
  f_\text{alias} = \frac{c}{2d_\min}
  \]
- Covariance matrix for subspace methods:
  \[
  R = E[x(t)x^H(t)]
  \]

### Example (16 mics, 8 cm aperture, 36 kHz bandwidth):
- Angular resolution ≈ **9°** at 36 kHz.
- Aliasing cutoff ≈ **8.6 kHz** (higher frequencies handled by DSP).

---

## 🧪 Simulation Tools
We provide Python prototypes for rapid testing.

### `geometry/fermat_spiral.py`
Plots the 16-mic Fermat spiral layout, annotates coordinates, and prints:
- Aperture diameter
- Angular resolution (degrees)
- Aliasing cutoff frequency (kHz)

### `dsp/fake_data.py`
Simulates real-time microphone data + beamforming visualization.
- Generates synthetic signals for one or more sources.
- Adds Gaussian noise.
- Performs delay-and-sum beamforming.
- Displays a **live beamforming plot** (animated).

---

## 🚀 Roadmap
- **Phase I (Fall semester)**  
  Literature review, requirements, mic selection, PCB/housing design, Python DSP prototypes.
- **Phase II (Winter semester)**  
  STM32 firmware, Pi integration, C++ DSP, MUSIC/ESPRIT implementation, real-time heatmaps, final validation.

---

## 👥 Team
- **David** – DSP lead, MUSIC/ESPRIT, visualization
- **Basem** – C++ DSP implementation, contingency software
- **Fabian** – Requirements, integration testing
- **Tim** – Literature review, peripheral testing
- **Rob** – Hardware selection, PCB design
- **Ahmad** – Housing, assembly, CAD

---

## 📚 References
- Fluke ii900 Acoustic Imager (industrial benchmark).
- Van Trees, *Detection, Estimation, and Modulation Theory* (classic array signal processing).
- Schmidt, “Multiple Emitter Location and Signal Parameter Estimation (MUSIC)” (1986).
- Roy & Kailath, “ESPRIT—Estimation of Signal Parameters via Rotational Invariance” (1989).

---

## 📝 License
This project is for academic purposes (COEN 490 Capstone).  
If you want to use it beyond this course, please contact the team for permissions.
