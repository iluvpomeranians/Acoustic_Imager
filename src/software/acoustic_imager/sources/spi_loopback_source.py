#!/usr/bin/env python3
from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from acoustic_imager import config
from acoustic_imager.custom_types import LatestFrame  # you already have this
# If LatestFrame.stats expects a specific stats type, import that too.
# Otherwise, we’ll create a small compatible stats object below.

@dataclass
class LoopbackStats:
    frames_ok: int = 0
    bad_parse: int = 0
    bad_crc: int = 0
    last_err: str = ""
    sclk_hz_rep: int = config.SPI_MAX_SPEED_HZ if hasattr(config, "SPI_MAX_SPEED_HZ") else 0


class SPILoopbackSource:
    """
    Pure-software 'SPI loopback' generator (no spidev, no STM32).

    Generates FFT frames (N_MICS x N_BINS complex64) that mimic the old
    v9 loopback behavior: a few deterministic virtual sources at chosen bins,
    with slight angle drift and spectral spreading.
    """

    def __init__(
        self,
        bins=(35, 80, 160, 220),
        ampls=(6.0, 3.0, 5.0, 4.0),
        angles_deg=(-25.0, 35.0, -5.0, 60.0),
        drift_deg_per_sec=(1.2, -0.6, 0.4, -0.3),
        spread_bins: int = 3,
        sigma_bins: float = 1.2,
        noise_mag: float = 0.006,
        noise_phase: float = 0.015,
        update_hz: float = 200.0,  # loopback generator rate (independent of UI FPS)
    ):
        self._bins = list(bins)
        self._ampls = list(ampls)
        self._angles = list(angles_deg)
        self._drift = list(drift_deg_per_sec)
        self._spread_bins = int(spread_bins)
        self._sigma_bins = float(sigma_bins)
        self._noise_mag = float(noise_mag)
        self._noise_phase = float(noise_phase)
        self._update_dt = 1.0 / float(update_hz)

        self._lock = threading.Lock()
        self._latest_fft: Optional[np.ndarray] = None
        self._latest_ok: bool = False
        self._frame_id: int = 0
        self._stop = True
        self._th: Optional[threading.Thread] = None

        self._stats = LoopbackStats()

    def start(self) -> None:
        if self._th is not None and self._th.is_alive():
            return
        self._stop = False
        self._th = threading.Thread(target=self._worker, daemon=True)
        self._th.start()

    def stop(self) -> None:
        self._stop = True
        # don’t join hard; keep it non-blocking
        time.sleep(0.01)

    def get_latest(self) -> LatestFrame:
        with self._lock:
            ok = self._latest_ok
            fft = self._latest_fft
            fid = self._frame_id
            stats = self._stats

        # LatestFrame in your code has fields: ok, fft_data, stats (based on usage)
        return LatestFrame(ok=ok, fft_data=fft, stats=stats, frame_id=fid)

    # -------------------------
    # Internal generation
    # -------------------------
    def _worker(self) -> None:
        t0 = time.time()
        while not self._stop:
            try:
                t = time.time() - t0
                fft = self._make_fft_frame(t)
                with self._lock:
                    self._frame_id += 1
                    self._latest_fft = fft
                    self._latest_ok = True
                    self._stats.frames_ok += 1
                    self._stats.last_err = ""
            except Exception as e:
                with self._lock:
                    self._latest_ok = False
                    self._stats.bad_parse += 1
                    self._stats.last_err = f"loop_exc:{e}"
            time.sleep(self._update_dt)

    def _make_fft_frame(self, t_sec: float) -> np.ndarray:
        """
        Returns fft_data (N_MICS x N_BINS) complex64
        """
        N_MICS = config.N_MICS
        N_BINS = config.N_BINS

        mp_mag = self._noise_mag * (1.0 + 0.25 * np.random.randn(N_MICS, N_BINS)).astype(np.float32)
        mp_ph  = (self._noise_phase * np.random.randn(N_MICS, N_BINS)).astype(np.float32)

        # Spread energy around bin centers
        for i, b0 in enumerate(self._bins):
            if not (0 <= b0 < N_BINS):
                continue

            f0 = float(config.f_axis[b0])

            ang = float(self._angles[i]) + float(self._drift[i]) * t_sec
            ang = ((ang + 90.0) % 180.0) - 90.0
            theta = np.deg2rad(ang)

            # steering vector a0 for this source
            a0 = np.exp(
                -1j * 2.0 * np.pi * f0 / config.SPEED_SOUND *
                -(config.x_coords * np.cos(theta) + config.y_coords * np.sin(theta))
            ).astype(np.complex64)

            for dbin in range(-self._spread_bins, self._spread_bins + 1):
                b = b0 + dbin
                if b < 0 or b >= N_BINS:
                    continue

                w = float(np.exp(-0.5 * (dbin / self._sigma_bins) ** 2))
                amp = float(self._ampls[i]) * w

                ph0 = 0.15 * dbin
                X = amp * a0 * np.exp(1j * ph0)

                mp_mag[:, b] += np.abs(X).astype(np.float32)
                mp_ph[:, b]  += np.angle(X).astype(np.float32)

        fft = (mp_mag * (np.cos(mp_ph) + 1j * np.sin(mp_ph))).astype(np.complex64)
        return fft
