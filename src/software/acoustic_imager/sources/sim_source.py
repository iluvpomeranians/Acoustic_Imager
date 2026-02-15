# sources/sim_source.py
from __future__ import annotations

import time
from typing import List

import numpy as np

import config
from sources.types import LatestFrame, SourceStats


def _cfg(name: str):
    if not hasattr(config, name):
        raise AttributeError(
            f"config.{name} is missing. sim_source expects it. "
            f"Add `{name} = ...` to config.py"
        )
    return getattr(config, name)


class SimSource:
    """
    Generates synthetic multi-source signals and returns FFT frames as LatestFrame.
    """

    def __init__(self) -> None:
        self.frame_id = 0

        self.N_MICS = int(_cfg("N_MICS"))
        self.SAMPLES_PER_CHANNEL = int(_cfg("SAMPLES_PER_CHANNEL"))
        self.SAMPLE_RATE_HZ = int(_cfg("SAMPLE_RATE_HZ"))
        self.SPEED_SOUND = float(_cfg("SPEED_SOUND"))
        self.NOISE_POWER = float(_cfg("NOISE_POWER"))

        self.x_coords = np.asarray(_cfg("x_coords"), dtype=np.float32)
        self.y_coords = np.asarray(_cfg("y_coords"), dtype=np.float32)

        self.SIM_SOURCE_FREQS: List[float] = list(_cfg("SIM_SOURCE_FREQS"))
        self.SIM_SOURCE_ANGLES: List[float] = list(_cfg("SIM_SOURCE_ANGLES"))
        self.SIM_SOURCE_AMPLS: List[float] = list(_cfg("SIM_SOURCE_AMPLS"))

        if not (len(self.SIM_SOURCE_FREQS) == len(self.SIM_SOURCE_ANGLES) == len(self.SIM_SOURCE_AMPLS)):
            raise ValueError("SIM_SOURCE_FREQS/ANGLES/AMPLS must be same length in config.py")

    def read_frame(self) -> LatestFrame:
        # small deterministic drift so it “moves”
        for k in range(len(self.SIM_SOURCE_ANGLES)):
            self.SIM_SOURCE_ANGLES[k] += (0.15 + 0.05 * k)
            if self.SIM_SOURCE_ANGLES[k] > 90.0:
                self.SIM_SOURCE_ANGLES[k] = -90.0

        self.frame_id += 1
        fft = self._generate_fft(self.SIM_SOURCE_ANGLES)

        stats = SourceStats(frames_ok=self.frame_id, last_err="", sclk_hz_rep=0)
        return LatestFrame(
            fft_data=fft,
            frame_id=self.frame_id,
            ok=True,
            stats=stats,
        )

    def _generate_fft(self, angle_degs: List[float]) -> np.ndarray:
        t = np.arange(self.SAMPLES_PER_CHANNEL, dtype=np.float32) / float(self.SAMPLE_RATE_HZ)
        mic_signals = np.zeros((self.N_MICS, t.size), dtype=np.float32)

        for src_idx, angle_deg in enumerate(angle_degs):
            angle_rad = np.deg2rad(float(angle_deg))
            f = float(self.SIM_SOURCE_FREQS[src_idx])
            amp = float(self.SIM_SOURCE_AMPLS[src_idx])

            cth = np.cos(angle_rad)
            sth = np.sin(angle_rad)

            # delays for all mics (vectorized)
            delays = -((self.x_coords * cth + self.y_coords * sth) / float(self.SPEED_SOUND))  # (M,)

            # signal per mic
            phase = 2.0 * np.pi * f * (t[None, :] - delays[:, None])
            mic_signals += amp * np.sin(phase).astype(np.float32)

        # noise
        if self.NOISE_POWER > 0:
            mic_signals += np.random.normal(0.0, np.sqrt(self.NOISE_POWER), mic_signals.shape).astype(np.float32)

        # FFT
        fft_data = np.fft.rfft(mic_signals, axis=1).astype(np.complex64)  # (M, N_BINS)
        return fft_data
