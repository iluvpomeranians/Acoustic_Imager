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
import threading
import struct
import zlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .. import config

# Optional spidev
try:
    import spidev  # type: ignore
    SPIDEV_AVAILABLE = True
except Exception:
    spidev = None  # type: ignore
    SPIDEV_AVAILABLE = False


# ===============================================================
# SPI framing helpers (same as monolith, moved here)
# ===============================================================
def crc_validate_frame(buf: bytes) -> bool:
    if len(buf) != config.FRAME_BYTES:
        return False
    try:
        (magic, ver, hdr_len, seq, mic, fft_size, fs, bins, _res, pay_len) = struct.unpack_from(
            config.HEADER_FMT, buf, 0
        )
    except struct.error:
        return False

    if magic != config.MAGIC_START or ver != config.VERSION or hdr_len != config.HEADER_LEN:
        return False
    if mic != config.N_MICS or fft_size != config.SAMPLES_PER_CHANNEL or fs != config.SAMPLE_RATE_HZ or bins != config.N_BINS:
        return False
    if pay_len != config.PAYLOAD_LEN:
        return False

    crc_rx, magic_end = struct.unpack_from(config.TRAILER_FMT, buf, config.HEADER_LEN + config.PAYLOAD_LEN)
    if magic_end != config.MAGIC_END:
        return False

    header = buf[: config.HEADER_LEN]
    payload = buf[config.HEADER_LEN : config.HEADER_LEN + config.PAYLOAD_LEN]
    crc = zlib.crc32(header)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return crc == crc_rx


def framing_validate_frame(buf: bytes) -> Tuple[bool, str]:
    if len(buf) != config.FRAME_BYTES:
        return False, "len"
    try:
        (magic,) = struct.unpack_from("<I", buf, 0)
    except struct.error:
        return False, "hdr_unpack0"
    if magic != config.MAGIC_START:
        return False, "magic_start"

    try:
        (magic, ver, hdr_len, seq, mic, fft_size, fs, bins, _res, pay_len) = struct.unpack_from(
            config.HEADER_FMT, buf, 0
        )
    except struct.error:
        return False, "hdr_unpack"

    if ver != config.VERSION or hdr_len != config.HEADER_LEN:
        return False, "hdr_fields"
    if mic != config.N_MICS or fft_size != config.SAMPLES_PER_CHANNEL or fs != config.SAMPLE_RATE_HZ or bins != config.N_BINS:
        return False, "cfg_mismatch"
    if pay_len != config.PAYLOAD_LEN:
        return False, "pay_len"

    try:
        (_, magic_end) = struct.unpack_from(config.TRAILER_FMT, buf, config.HEADER_LEN + config.PAYLOAD_LEN)
    except struct.error:
        return False, "trl_unpack"
    if magic_end != config.MAGIC_END:
        return False, "magic_end"

    return True, "ok"


def make_payload_mag_phase(t_sec: float) -> bytes:
    """
    Loopback payload with stable virtual sources (+ bin spreading), same as monolith.
    Returns bytes of float32 array shaped (N_MICS, N_BINS, 2): [MAG, PHASE].
    """
    mp = np.zeros((config.N_MICS, config.N_BINS, 2), dtype=np.float32)

    noise_mag = 0.006
    noise_phase = 0.015
    mp[:, :, 0] = noise_mag * (1.0 + 0.25 * np.random.randn(config.N_MICS, config.N_BINS)).astype(np.float32)
    mp[:, :, 1] = noise_phase * np.random.randn(config.N_MICS, config.N_BINS).astype(np.float32)

    spread_bins = 3
    sigma_bins = 1.2

    for i, b0 in enumerate(config.SPI_SIM_BINS):
        if b0 < 0 or b0 >= config.N_BINS:
            continue
        if i >= len(config.SPI_SIM_AMPLS) or i >= len(config.SPI_SIM_ANGLES) or i >= len(config.SPI_SIM_DRIFT_DEG_PER_SEC):
            continue

        f0 = float(config.f_axis[b0])
        ang = config.SPI_SIM_ANGLES[i] + config.SPI_SIM_DRIFT_DEG_PER_SEC[i] * float(t_sec)
        ang = ((ang + 90.0) % 180.0) - 90.0
        theta = np.deg2rad(ang)

        a0 = np.exp(
            -1j * 2 * np.pi * f0 / config.SPEED_SOUND
            * -(config.x_coords * np.cos(theta) + config.y_coords * np.sin(theta))
        ).astype(np.complex64)

        for dbin in range(-spread_bins, spread_bins + 1):
            b = b0 + dbin
            if b < 0 or b >= config.N_BINS:
                continue

            w = float(np.exp(-0.5 * (dbin / sigma_bins) ** 2))
            amp = float(config.SPI_SIM_AMPLS[i]) * w

            ph0 = 0.15 * dbin
            X = amp * a0 * np.exp(1j * ph0)

            mp[:, b, 0] += np.abs(X).astype(np.float32)
            mp[:, b, 1] += np.angle(X).astype(np.float32)

    return mp.tobytes(order="C")


def make_frame(seq: int, t_sec: float) -> bytes:
    payload = make_payload_mag_phase(t_sec)
    header = struct.pack(
        config.HEADER_FMT,
        config.MAGIC_START,
        config.VERSION,
        config.HEADER_LEN,
        int(seq),
        config.N_MICS,
        config.SAMPLES_PER_CHANNEL,
        config.SAMPLE_RATE_HZ,
        config.N_BINS,
        0,
        len(payload),
    )
    crc = zlib.crc32(header)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    trailer = struct.pack(config.TRAILER_FMT, crc, config.MAGIC_END)
    return header + payload + trailer


def spi_xfer_bytes(spi_dev, tx: bytes) -> bytes:
    rx = bytearray(len(tx))
    mv = memoryview(tx)
    offset = 0
    while offset < len(tx):
        end = min(offset + config.SPI_XFER_CHUNK, len(tx))
        chunk = mv[offset:end]
        r = spi_dev.xfer3(chunk)
        rx[offset:end] = bytes(r)
        offset = end
    return bytes(rx)


# ===============================================================
# SPI Source (open/read/close) - same behavior as monolith
# ===============================================================
class SpiFFTSource:
    def __init__(self, bus: int = config.SPI_BUS, dev: int = config.SPI_DEV, max_speed_hz: int = config.SPI_MAX_SPEED_HZ):
        self.bus = int(bus)
        self.dev = int(dev)
        self.max_speed_hz = int(max_speed_hz)
        self.spi = None

        self.frames_ok = 0
        self.bad_parse = 0
        self.bad_crc = 0
        self.last_err = ""
        self.seq_seen = 0
        self.sclk_hz_rep = 0

    def open(self) -> bool:
        if not SPIDEV_AVAILABLE:
            self.last_err = "spidev not installed"
            return False
        try:
            self.spi = spidev.SpiDev()  # type: ignore
            self.spi.open(self.bus, self.dev)
            self.spi.mode = config.SPI_MODE
            self.spi.bits_per_word = config.SPI_BITS
            self.spi.max_speed_hz = int(self.max_speed_hz)
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

    def read_fft_data(self) -> Optional[np.ndarray]:
        """
        Returns fft_data as complex64 array shaped (N_MICS, N_BINS), or None on failure.
        """
        if self.spi is None:
            if not self.open():
                return None

        try:
            self.seq_seen += 1
            t_sec = time.time()

            tx = make_frame(self.seq_seen, t_sec)
            rx = spi_xfer_bytes(self.spi, tx)

            nz = int(np.count_nonzero(np.frombuffer(rx, dtype=np.uint8, count=256)))
            self.last_err = f"rx_nonzero256={nz}"

            ok, why = framing_validate_frame(rx)
            if not ok:
                self.bad_parse += 1
                self.last_err = f"parse:{why} nz256={nz}"
                return None

            if config.CRC_EVERY_N and ((self.frames_ok % config.CRC_EVERY_N) == 0):
                if not crc_validate_frame(rx):
                    self.bad_crc += 1
                    self.last_err = "crc"
                    return None

            payload = rx[config.HEADER_LEN : config.HEADER_LEN + config.PAYLOAD_LEN]
            mp = np.frombuffer(payload, dtype=np.float32).reshape(config.N_MICS, config.N_BINS, 2)
            mag = mp[:, :, 0]
            phase = mp[:, :, 1]
            fft_data = (mag * (np.cos(phase) + 1j * np.sin(phase))).astype(np.complex64)

            self.frames_ok += 1
            return fft_data

        except Exception as e:
            self.bad_parse += 1
            self.last_err = f"spi_exc:{e}"
            return None


# ===============================================================
# Threaded SPI Manager (mailbox like monolith _LatestSpiData)
# ===============================================================
@dataclass
class LatestSpiData:
    lock: threading.Lock
    fft_data: Optional[np.ndarray] = None
    ok: bool = False
    frame_id: int = 0
    last_err: str = ""
    frames_ok: int = 0
    bad_parse: int = 0
    bad_crc: int = 0
    sclk_hz_rep: int = 0


class SpiManager:
    """
    Starts a worker that continuously reads SPI and stores ONLY the latest fft_data.
    Main/UI can call get_latest() without blocking.
    """

    def __init__(self, source: Optional[SpiFFTSource] = None) -> None:
        self.source = source if source is not None else SpiFFTSource()

        self._latest = LatestSpiData(lock=threading.Lock())
        self._stop = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop = False
        with self._latest.lock:
            self._latest.ok = False
            self._latest.fft_data = None
            self._latest.frame_id = 0
            self._latest.last_err = ""

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        time.sleep(0.01)
        self._thread = None
        self.source.close()

    def get_latest(self) -> Tuple[Optional[np.ndarray], str]:
        """
        Returns (fft_data, last_err).
        """
        with self._latest.lock:
            return self._latest.fft_data, self._latest.last_err

    def copy_stats_to(self, target) -> None:
        """
        Convenience: mirror stats into another object (e.g., your UI wants spi_source.frames_ok, etc).
        """
        with self._latest.lock:
            target.frames_ok = self._latest.frames_ok
            target.bad_parse = self._latest.bad_parse
            target.bad_crc = self._latest.bad_crc
            target.sclk_hz_rep = self._latest.sclk_hz_rep
            target.last_err = self._latest.last_err

    def _worker(self) -> None:
        while not self._stop:
            try:
                fft = self.source.read_fft_data()

                with self._latest.lock:
                    self._latest.last_err = self.source.last_err
                    self._latest.frames_ok = self.source.frames_ok
                    self._latest.bad_parse = self.source.bad_parse
                    self._latest.bad_crc = self.source.bad_crc
                    self._latest.sclk_hz_rep = self.source.sclk_hz_rep

                    if fft is not None:
                        self._latest.fft_data = fft
                        self._latest.frame_id += 1
                        self._latest.ok = True

            except Exception as e:
                with self._latest.lock:
                    self._latest.last_err = f"spi_thread_exc:{e}"
                    self._latest.bad_parse = self.source.bad_parse
                time.sleep(0.002)
