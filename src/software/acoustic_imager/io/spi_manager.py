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
from dataclasses import replace
from typing import Optional

import numpy as np

from sources.types import LatestFrame, SourceStats

try:
    import spidev  # type: ignore
    SPIDEV_AVAILABLE = True
except Exception:
    spidev = None
    SPIDEV_AVAILABLE = False

# Import config as a module of constants.
# Expected keys are listed in _cfg() below.
import config


def _cfg(name: str):
    if not hasattr(config, name):
        raise AttributeError(
            f"config.{name} is missing. spi_manager expects it. "
            f"Add `{name} = ...` to config.py"
        )
    return getattr(config, name)


class SPIManager:
    """
    Threaded SPI reader. Stores "latest FFT frame" in a mailbox (LatestFrame).
    Supports your STM32 framing format: header + payload(mag/phase float32) + crc32 + end magic.

    Public API:
      - start()
      - stop()
      - get_latest() -> LatestFrame (safe copy)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest = LatestFrame()

        self._stop = False
        self._thread: Optional[threading.Thread] = None

        self._spi = None
        self._seq_seen = 0

        # --- framing constants from config ---
        self.N_MICS = int(_cfg("N_MICS"))
        self.SAMPLES_PER_CHANNEL = int(_cfg("SAMPLES_PER_CHANNEL"))
        self.SAMPLE_RATE_HZ = int(_cfg("SAMPLE_RATE_HZ"))
        self.N_BINS = int(_cfg("N_BINS"))

        self.MAGIC_START = int(_cfg("MAGIC_START"))
        self.MAGIC_END = int(_cfg("MAGIC_END"))
        self.VERSION = int(_cfg("VERSION"))

        self.HEADER_FMT = str(_cfg("HEADER_FMT"))
        self.HEADER_LEN = int(_cfg("HEADER_LEN"))
        self.TRAILER_FMT = str(_cfg("TRAILER_FMT"))
        self.TRAILER_LEN = int(_cfg("TRAILER_LEN"))
        self.PAYLOAD_LEN = int(_cfg("PAYLOAD_LEN"))
        self.FRAME_BYTES = int(_cfg("FRAME_BYTES"))

        # --- spi config ---
        self.SPI_BUS = int(_cfg("SPI_BUS"))
        self.SPI_DEV = int(_cfg("SPI_DEV"))
        self.SPI_MODE = int(_cfg("SPI_MODE"))
        self.SPI_BITS = int(_cfg("SPI_BITS"))
        self.SPI_MAX_SPEED_HZ = int(_cfg("SPI_MAX_SPEED_HZ"))
        self.SPI_XFER_CHUNK = int(_cfg("SPI_XFER_CHUNK"))

        # how often to CRC check (0 = never)
        self.CRC_EVERY_N = int(getattr(config, "CRC_EVERY_N", 30))

    # ------------------------------
    # Public API
    # ------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        time.sleep(0.01)
        self._close_spi()

    def get_latest(self) -> LatestFrame:
        # Return a safe copy: copy fft array ref (ok), and copy stats values
        with self._lock:
            lf = self._latest
            stats_copy = replace(lf.stats)
            out = LatestFrame(
                fft_data=lf.fft_data,
                frame_id=int(lf.frame_id),
                ok=bool(lf.ok),
                stats=stats_copy,
            )
        return out

    # ------------------------------
    # SPI device open/close
    # ------------------------------
    def _open_spi(self) -> bool:
        if not SPIDEV_AVAILABLE:
            self._set_err("spidev not installed")
            return False
        try:
            self._spi = spidev.SpiDev()
            self._spi.open(self.SPI_BUS, self.SPI_DEV)
            self._spi.mode = self.SPI_MODE
            self._spi.bits_per_word = self.SPI_BITS
            self._spi.max_speed_hz = int(self.SPI_MAX_SPEED_HZ)
            self._set_sclk(int(self._spi.max_speed_hz))
            self._set_err("")
            return True
        except Exception as e:
            self._spi = None
            self._set_err(f"SPI open failed: {e}")
            return False

    def _close_spi(self) -> None:
        if self._spi is not None:
            try:
                self._spi.close()
            except Exception:
                pass
        self._spi = None

    # ------------------------------
    # Worker loop
    # ------------------------------
    def _worker(self) -> None:
        # reset mailbox
        with self._lock:
            self._latest = LatestFrame()

        while not self._stop:
            if self._spi is None:
                if not self._open_spi():
                    time.sleep(0.1)
                    continue

            fr = self._read_one()
            if fr is not None:
                with self._lock:
                    self._latest = fr

    # ------------------------------
    # One frame read
    # ------------------------------
    def _read_one(self) -> Optional[LatestFrame]:
        self._seq_seen += 1

        try:
            # IMPORTANT:
            # This assumes your STM32 returns bytes in same transfer (full duplex).
            # If your STM32 is real, you likely send dummy bytes and read real bytes.
            # Here we just send zeros of FRAME_BYTES and read back FRAME_BYTES.
            tx = bytes(self.FRAME_BYTES)
            rx = self._spi_xfer_bytes(tx)

            ok, why = self._framing_validate(rx)
            stats = self._get_stats()
            if not ok:
                stats.bad_parse += 1
                stats.last_err = f"parse:{why}"
                self._set_stats(stats)
                return None

            if self.CRC_EVERY_N and ((stats.frames_ok % self.CRC_EVERY_N) == 0):
                if not self._crc_validate(rx):
                    stats.bad_crc += 1
                    stats.last_err = "crc"
                    self._set_stats(stats)
                    return None

            payload = rx[self.HEADER_LEN : self.HEADER_LEN + self.PAYLOAD_LEN]
            mp = np.frombuffer(payload, dtype=np.float32).reshape(self.N_MICS, self.N_BINS, 2)

            mag = mp[:, :, 0]
            phase = mp[:, :, 1]
            fft_data = (mag * (np.cos(phase) + 1j * np.sin(phase))).astype(np.complex64)

            stats.frames_ok += 1
            stats.last_err = ""
            self._set_stats(stats)

            return LatestFrame(
                fft_data=fft_data,
                frame_id=self._seq_seen,
                ok=True,
                stats=stats,
            )

        except Exception as e:
            stats = self._get_stats()
            stats.bad_parse += 1
            stats.last_err = f"spi_exc:{e}"
            self._set_stats(stats)
            return None

    # ------------------------------
    # Transfer
    # ------------------------------
    def _spi_xfer_bytes(self, tx: bytes) -> bytes:
        assert self._spi is not None
        rx = bytearray(len(tx))
        mv = memoryview(tx)
        offset = 0

        while offset < len(tx):
            end = min(offset + self.SPI_XFER_CHUNK, len(tx))
            chunk = mv[offset:end]
            r = self._spi.xfer3(chunk)
            rx[offset:end] = bytes(r)
            offset = end

        return bytes(rx)

    # ------------------------------
    # Framing validation
    # ------------------------------
    def _framing_validate(self, buf: bytes) -> tuple[bool, str]:
        if len(buf) != self.FRAME_BYTES:
            return False, "len"
        try:
            (magic,) = struct.unpack_from("<I", buf, 0)
        except struct.error:
            return False, "hdr_unpack0"
        if magic != self.MAGIC_START:
            return False, "magic_start"

        try:
            (magic, ver, hdr_len, seq, mic, fft_size, fs, bins, _res, pay_len) = struct.unpack_from(
                self.HEADER_FMT, buf, 0
            )
        except struct.error:
            return False, "hdr_unpack"

        if ver != self.VERSION or hdr_len != self.HEADER_LEN:
            return False, "hdr_fields"
        if mic != self.N_MICS or fft_size != self.SAMPLES_PER_CHANNEL or fs != self.SAMPLE_RATE_HZ or bins != self.N_BINS:
            return False, "cfg_mismatch"
        if pay_len != self.PAYLOAD_LEN:
            return False, "pay_len"

        try:
            (_, magic_end) = struct.unpack_from(self.TRAILER_FMT, buf, self.HEADER_LEN + self.PAYLOAD_LEN)
        except struct.error:
            return False, "trl_unpack"
        if magic_end != self.MAGIC_END:
            return False, "magic_end"

        return True, "ok"

    def _crc_validate(self, buf: bytes) -> bool:
        if len(buf) != self.FRAME_BYTES:
            return False

        try:
            (magic, ver, hdr_len, seq, mic, fft_size, fs, bins, _res, pay_len) = struct.unpack_from(
                self.HEADER_FMT, buf, 0
            )
        except struct.error:
            return False

        if magic != self.MAGIC_START or ver != self.VERSION or hdr_len != self.HEADER_LEN:
            return False
        if mic != self.N_MICS or fft_size != self.SAMPLES_PER_CHANNEL or fs != self.SAMPLE_RATE_HZ or bins != self.N_BINS:
            return False
        if pay_len != self.PAYLOAD_LEN:
            return False

        crc_rx, magic_end = struct.unpack_from(self.TRAILER_FMT, buf, self.HEADER_LEN + self.PAYLOAD_LEN)
        if magic_end != self.MAGIC_END:
            return False

        header = buf[: self.HEADER_LEN]
        payload = buf[self.HEADER_LEN : self.HEADER_LEN + self.PAYLOAD_LEN]
        crc = zlib.crc32(header)
        crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
        return crc == crc_rx

    # ------------------------------
    # Stats helpers (thread-safe)
    # ------------------------------
    def _get_stats(self) -> SourceStats:
        with self._lock:
            return replace(self._latest.stats)

    def _set_stats(self, stats: SourceStats) -> None:
        with self._lock:
            self._latest.stats = stats

    def _set_err(self, msg: str) -> None:
        stats = self._get_stats()
        stats.last_err = msg
        self._set_stats(stats)

    def _set_sclk(self, hz: int) -> None:
        stats = self._get_stats()
        stats.sclk_hz_rep = int(hz)
        self._set_stats(stats)
