#!/usr/bin/env python3
"""
spi_manager.py

Manages SPI acquisition for the Acoustic Imager (loopback or real device).

Responsibilities:
- Defines the SPI frame format (header/payload/trailer) and validation helpers.
- Generates loopback frames (mag/phase) when operating without an STM32.
- Provides a SpiFFTSource reader that produces FFTFrame-like data (complex64 bins).
- Runs SPI reading in a background worker thread and exposes the latest fft_data + stats.
"""

from __future__ import annotations

import time
import struct
import zlib
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# spidev optional
try:
    import spidev  # type: ignore
    SPIDEV_AVAILABLE = True
except Exception:
    spidev = None
    SPIDEV_AVAILABLE = False


# ----------------------------
# Lightweight FFT frame holder
# ----------------------------
@dataclass
class SpiFFTFrame:
    channel_count: int
    sampling_rate: int
    fft_size: int
    frame_id: int
    fft_data: np.ndarray  # shape: (N_MICS, N_BINS), dtype=complex64


@dataclass
class SpiConfig:
    n_mics: int
    samples_per_channel: int
    sample_rate_hz: int
    n_bins: int
    f_axis: np.ndarray

    x_coords: np.ndarray
    y_coords: np.ndarray
    speed_sound: float

    # SPI HW config
    bus: int
    dev: int
    mode: int
    bits: int
    max_speed_hz: int
    xfer_chunk: int

    # Frame format
    magic_start: int
    magic_end: int
    version: int
    header_fmt: str
    trailer_fmt: str

    crc_every_n: int

    # Loopback "virtual sources"
    sim_bins: list
    sim_ampls: list
    sim_angles: list
    sim_drift_deg_per_sec: list
    spread_bins: int = 3
    sigma_bins: float = 1.2


class SpiManager:
    """
    Combines:
      - SpiFFTSource (blocking read_frame)
      - background latest-data thread (non-blocking main loop access)
      - stats mirroring: ok frames, bad parse, bad crc, last_err, sclk_hz_rep
    """

    def __init__(self, cfg: SpiConfig):
        self.cfg = cfg
        self.source = SpiFFTSource(cfg)

        self._lock = threading.Lock()
        self._latest_fft: Optional[np.ndarray] = None
        self._latest_ok: bool = False
        self._latest_frame_id: int = 0

        self._last_err: str = ""
        self._frames_ok: int = 0
        self._bad_parse: int = 0
        self._bad_crc: int = 0
        self._sclk_hz_rep: int = 0

        self._stop = False
        self._thread: Optional[threading.Thread] = None

    # ----------------------------
    # Thread control
    # ----------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        time.sleep(0.02)
        self._thread = None
        self.source.close()
        with self._lock:
            self._latest_fft = None
            self._latest_ok = False

    # ----------------------------
    # Non-blocking access
    # ----------------------------
    def get_latest(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            return self._latest_ok, self._latest_fft

    def get_stats(self) -> Tuple[int, int, int, int, str]:
        """
        Returns: (frames_ok, bad_parse, bad_crc, sclk_hz_rep, last_err)
        """
        with self._lock:
            return self._frames_ok, self._bad_parse, self._bad_crc, self._sclk_hz_rep, self._last_err

    # ----------------------------
    # Worker loop
    # ----------------------------
    def _worker(self) -> None:
        with self._lock:
            self._latest_ok = False
            self._latest_fft = None
            self._latest_frame_id = 0
            self._last_err = ""

        while not self._stop:
            try:
                fr = self.source.read_frame()

                with self._lock:
                    # mirror stats
                    self._last_err = self.source.last_err
                    self._frames_ok = self.source.frames_ok
                    self._bad_parse = self.source.bad_parse
                    self._bad_crc = self.source.bad_crc
                    self._sclk_hz_rep = self.source.sclk_hz_rep

                    if fr is not None and fr.fft_data is not None:
                        self._latest_fft = fr.fft_data
                        self._latest_frame_id = fr.frame_id
                        self._latest_ok = True

            except Exception as e:
                with self._lock:
                    self._last_err = f"spi_thread_exc:{e}"
                    self._bad_parse = self.source.bad_parse
                time.sleep(0.002)


# ===============================================================
# SPI source + framing helpers (from monolithic version)
# ===============================================================

class SpiFFTSource:
    def __init__(self, cfg: SpiConfig):
        self.cfg = cfg
        self.spi = None

        self.frames_ok = 0
        self.bad_parse = 0
        self.bad_crc = 0
        self.last_err = ""
        self.seq_seen = 0
        self.sclk_hz_rep = 0

        self.header_len = struct.calcsize(cfg.header_fmt)
        self.trailer_len = struct.calcsize(cfg.trailer_fmt)
        self.payload_len = cfg.n_mics * cfg.n_bins * 2 * 4
        self.frame_bytes = self.header_len + self.payload_len + self.trailer_len

    def open(self) -> bool:
        if not SPIDEV_AVAILABLE:
            self.last_err = "spidev not installed"
            return False
        try:
            self.spi = spidev.SpiDev()
            self.spi.open(self.cfg.bus, self.cfg.dev)
            self.spi.mode = self.cfg.mode
            self.spi.bits_per_word = self.cfg.bits
            self.spi.max_speed_hz = int(self.cfg.max_speed_hz)
            self.sclk_hz_rep = int(self.spi.max_speed_hz)
            self.last_err = ""
            return True
        except Exception as e:
            self.last_err = f"SPI open failed: {e}"
            self.spi = None
            return False

    def close(self) -> None:
        if self.spi is not None:
            try:
                self.spi.close()
            except Exception:
                pass
        self.spi = None

    def read_frame(self) -> Optional[SpiFFTFrame]:
        if self.spi is None:
            if not self.open():
                return None

        try:
            self.seq_seen += 1
            t_sec = time.time()

            tx = make_frame(self.cfg, self.seq_seen, t_sec)
            rx = spi_xfer_bytes(self.cfg, self.spi, tx)

            # quick debug info
            nz = int(np.count_nonzero(np.frombuffer(rx, dtype=np.uint8, count=min(256, len(rx)))))
            self.last_err = f"rx_nonzero256={nz}"

            ok, why = framing_validate_frame(self.cfg, rx)
            if not ok:
                self.bad_parse += 1
                self.last_err = f"parse:{why} nz256={nz}"
                return None

            if self.cfg.crc_every_n and ((self.frames_ok % self.cfg.crc_every_n) == 0):
                if not crc_validate_frame(self.cfg, rx):
                    self.bad_crc += 1
                    self.last_err = "crc"
                    return None

            payload = rx[self.header_len:self.header_len + self.payload_len]
            mp = np.frombuffer(payload, dtype=np.float32).reshape(self.cfg.n_mics, self.cfg.n_bins, 2)
            mag = mp[:, :, 0]
            phase = mp[:, :, 1]
            fft_data = (mag * (np.cos(phase) + 1j * np.sin(phase))).astype(np.complex64)

            out = SpiFFTFrame(
                channel_count=self.cfg.n_mics,
                sampling_rate=self.cfg.sample_rate_hz,
                fft_size=self.cfg.samples_per_channel,
                frame_id=self.seq_seen,
                fft_data=fft_data,
            )

            self.frames_ok += 1
            return out

        except Exception as e:
            self.bad_parse += 1
            self.last_err = f"spi_exc:{e}"
            return None


def crc_validate_frame(cfg: SpiConfig, buf: bytes) -> bool:
    header_len = struct.calcsize(cfg.header_fmt)
    trailer_len = struct.calcsize(cfg.trailer_fmt)
    payload_len = cfg.n_mics * cfg.n_bins * 2 * 4
    frame_bytes = header_len + payload_len + trailer_len

    if len(buf) != frame_bytes:
        return False

    try:
        fields = struct.unpack_from(cfg.header_fmt, buf, 0)
    except struct.error:
        return False

    magic = fields[0]
    ver = fields[1]
    hdr_len = fields[2]
    mic = fields[4]
    fft_size = fields[5]
    fs = fields[6]
    bins = fields[7]
    pay_len = fields[9]

    if magic != cfg.magic_start or ver != cfg.version or hdr_len != header_len:
        return False
    if mic != cfg.n_mics or fft_size != cfg.samples_per_channel or fs != cfg.sample_rate_hz or bins != cfg.n_bins:
        return False
    if pay_len != payload_len:
        return False

    crc_rx, magic_end = struct.unpack_from(cfg.trailer_fmt, buf, header_len + payload_len)
    if magic_end != cfg.magic_end:
        return False

    header = buf[:header_len]
    payload = buf[header_len:header_len + payload_len]
    crc = zlib.crc32(header)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return crc == crc_rx


def framing_validate_frame(cfg: SpiConfig, buf: bytes) -> Tuple[bool, str]:
    header_len = struct.calcsize(cfg.header_fmt)
    trailer_len = struct.calcsize(cfg.trailer_fmt)
    payload_len = cfg.n_mics * cfg.n_bins * 2 * 4
    frame_bytes = header_len + payload_len + trailer_len

    if len(buf) != frame_bytes:
        return False, "len"

    try:
        (magic,) = struct.unpack_from("<I", buf, 0)
    except struct.error:
        return False, "hdr_unpack0"

    if magic != cfg.magic_start:
        return False, "magic_start"

    try:
        fields = struct.unpack_from(cfg.header_fmt, buf, 0)
    except struct.error:
        return False, "hdr_unpack"

    ver = fields[1]
    hdr_len = fields[2]
    mic = fields[4]
    fft_size = fields[5]
    fs = fields[6]
    bins = fields[7]
    pay_len = fields[9]

    if ver != cfg.version or hdr_len != header_len:
        return False, "hdr_fields"
    if mic != cfg.n_mics or fft_size != cfg.samples_per_channel or fs != cfg.sample_rate_hz or bins != cfg.n_bins:
        return False, "cfg_mismatch"
    if pay_len != payload_len:
        return False, "pay_len"

    try:
        (_, magic_end) = struct.unpack_from(cfg.trailer_fmt, buf, header_len + payload_len)
    except struct.error:
        return False, "trl_unpack"

    if magic_end != cfg.magic_end:
        return False, "magic_end"

    return True, "ok"


def make_payload_mag_phase(cfg: SpiConfig, seq: int, t_sec: float) -> bytes:
    mp = np.zeros((cfg.n_mics, cfg.n_bins, 2), dtype=np.float32)

    # base noise (matches monolithic "feel")
    noise_mag = 0.006
    noise_phase = 0.015
    mp[:, :, 0] = noise_mag * (1.0 + 0.25 * np.random.randn(cfg.n_mics, cfg.n_bins)).astype(np.float32)
    mp[:, :, 1] = noise_phase * np.random.randn(cfg.n_mics, cfg.n_bins).astype(np.float32)

    spread_bins = int(cfg.spread_bins)
    sigma_bins = float(cfg.sigma_bins)

    for i, b0 in enumerate(cfg.sim_bins):
        if b0 < 0 or b0 >= cfg.n_bins:
            continue
        if i >= len(cfg.sim_ampls) or i >= len(cfg.sim_angles) or i >= len(cfg.sim_drift_deg_per_sec):
            continue

        f0 = float(cfg.f_axis[b0])
        ang = cfg.sim_angles[i] + cfg.sim_drift_deg_per_sec[i] * t_sec
        ang = ((ang + 90.0) % 180.0) - 90.0
        theta = np.deg2rad(ang)

        a0 = np.exp(
            -1j * 2 * np.pi * f0 / cfg.speed_sound *
            -(cfg.x_coords * np.cos(theta) + cfg.y_coords * np.sin(theta))
        ).astype(np.complex64)

        for dbin in range(-spread_bins, spread_bins + 1):
            b = b0 + dbin
            if b < 0 or b >= cfg.n_bins:
                continue

            w = float(np.exp(-0.5 * (dbin / sigma_bins) ** 2))
            amp = float(cfg.sim_ampls[i]) * w
            ph0 = 0.15 * dbin
            X = amp * a0 * np.exp(1j * ph0)

            mp[:, b, 0] += np.abs(X).astype(np.float32)
            mp[:, b, 1] += np.angle(X).astype(np.float32)

    return mp.tobytes(order="C")


def make_frame(cfg: SpiConfig, seq: int, t_sec: float) -> bytes:
    header_len = struct.calcsize(cfg.header_fmt)
    payload = make_payload_mag_phase(cfg, seq, t_sec)
    header = struct.pack(
        cfg.header_fmt,
        cfg.magic_start, cfg.version, header_len, seq,
        cfg.n_mics, cfg.samples_per_channel, cfg.sample_rate_hz,
        cfg.n_bins, 0, len(payload),
    )
    crc = zlib.crc32(header)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    trailer = struct.pack(cfg.trailer_fmt, crc, cfg.magic_end)
    return header + payload + trailer


def spi_xfer_bytes(cfg: SpiConfig, spi, tx: bytes) -> bytes:
    rx = bytearray(len(tx))
    mv = memoryview(tx)
    offset = 0

    while offset < len(tx):
        end = min(offset + int(cfg.xfer_chunk), len(tx))
        chunk = mv[offset:end]
        r = spi.xfer3(chunk)
        rx[offset:end] = bytes(r)
        offset = end

    return bytes(rx)
