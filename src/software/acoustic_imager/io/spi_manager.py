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

from acoustic_imager.custom_types import LatestFrame, SourceStats
from acoustic_imager.spi.frame_ready import FrameReady, FrameReadyGPIO
from acoustic_imager.spi.spi_protocol import parse_mic_packet

try:
    import spidev
    SPIDEV_AVAILABLE = True
except Exception:
    spidev = None
    SPIDEV_AVAILABLE = False

from acoustic_imager import config

def _cfg(name: str):
    if not hasattr(config, name):
        raise AttributeError(
            f"config.{name} is missing. spi_manager expects it. "
            f"Add `{name} = ...` to config.py"
        )
    return getattr(config, name)


class BatchAccumulator:
    """
    Accumulates per-mic packets by batch_id; returns a LatestFrame when all N_MICS
    for a batch have been received. Thread-safe when used under the same lock as _latest.
    """
    def __init__(self, n_mics: int, n_bins: int) -> None:
        self._n_mics = n_mics
        self._n_bins = n_bins
        self._batches: dict[int, tuple[np.ndarray, set[int]]] = {}
        self._max_batches = 4  # cap to avoid unbounded growth

    def add_mic(
        self,
        batch_id: int,
        mic_index: int,
        fft_1mic: np.ndarray,
        stats: SourceStats,
    ) -> Optional[LatestFrame]:
        if mic_index < 0 or mic_index >= self._n_mics:
            return None
        if batch_id not in self._batches:
            if len(self._batches) >= self._max_batches:
                # drop oldest batch (arbitrary: min key)
                old = min(self._batches.keys())
                del self._batches[old]
            self._batches[batch_id] = (
                np.zeros((self._n_mics, self._n_bins), dtype=np.complex64),
                set(),
            )
        arr, received = self._batches[batch_id]
        arr[mic_index, :] = fft_1mic
        received.add(mic_index)
        if len(received) == self._n_mics:
            frame = LatestFrame(
                fft_data=arr.copy(),
                frame_id=batch_id,
                ok=True,
                stats=stats,
            )
            del self._batches[batch_id]
            return frame
        return None  # no full batch yet


class SPIManager:
    """
    Threaded SPI reader. Stores "latest FFT frame" in a mailbox (LatestFrame).
    Supports your STM32 framing format: header + payload(mag/phase float32) + crc32 + end magic.

    Public API:
      - start()
      - stop()
      - get_latest() -> LatestFrame (safe copy)
    """

    def __init__(
        self,
        frame_ready: Optional[FrameReady] = None,
        use_frame_ready: bool = True,
    ) -> None:
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

        # --- per-mic packet (firmware format) ---
        self.SPI_MIC_PACKET_BYTES = int(getattr(config, "SPI_MIC_PACKET_BYTES", 2081))
        self._accumulator = BatchAccumulator(self.N_MICS, self.N_BINS)

        # --- spi config ---
        self.SPI_BUS = int(_cfg("SPI_BUS"))
        self.SPI_DEV = int(_cfg("SPI_DEV"))
        self.SPI_MODE = int(_cfg("SPI_MODE"))
        self.SPI_BITS = int(_cfg("SPI_BITS"))
        self.SPI_MAX_SPEED_HZ = int(_cfg("SPI_MAX_SPEED_HZ"))
        self.SPI_XFER_CHUNK = int(_cfg("SPI_XFER_CHUNK"))

        # how often to CRC check (0 = never)
        self.CRC_EVERY_N = int(getattr(config, "CRC_EVERY_N", 30))

        # --- frame-ready (GPIO IRQ) ---
        self.use_frame_ready = bool(use_frame_ready)
        self._frame_ready: Optional[FrameReady] = frame_ready

    # ------------------------------
    # Public API
    # ------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        # Create FrameReadyGPIO only when enabled (SPI_HW)
        if self.use_frame_ready and self._frame_ready is None:
            bcm_pin = int(getattr(config, "FRAME_READY_BCM_PIN", 25))   # GPIO25 (BCM) == physical pin 22
            pull = getattr(config, "FRAME_READY_PULL", "down")          # "up" or "down"
            self._frame_ready = FrameReadyGPIO(bcm_pin=bcm_pin, pull=pull)

        self._stop = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        self._thread = None

        self._close_spi()

        if self._frame_ready is not None:
            try:
                self._frame_ready.close()
            except Exception:
                pass
            self._frame_ready = None


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
        with self._lock:
            self._latest = LatestFrame()
        self._accumulator = BatchAccumulator(self.N_MICS, self.N_BINS)

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
    # One mic packet read; accumulate until full batch
    # ------------------------------
    def _read_one(self) -> Optional[LatestFrame]:
        # Wait for STM32 "frame ready" pulse (one mic packet ready)
        if self.use_frame_ready and self._frame_ready is not None:
            timeout_s = float(getattr(config, "FRAME_READY_TIMEOUT_S", 0.25))
            self._frame_ready.clear()
            got = self._frame_ready.wait(timeout=timeout_s)
            if not got:
                return None

        try:
            tx = bytes(self.SPI_MIC_PACKET_BYTES)
            rx = self._spi_xfer_bytes(tx)

            ok, batch_id, mic_index, fft_1mic = parse_mic_packet(rx)
            stats = self._get_stats()
            if not ok:
                stats.bad_parse += 1
                stats.last_err = "parse:mic_packet"
                self._set_stats(stats)
                return None

            frame = self._accumulator.add_mic(batch_id, mic_index, fft_1mic, stats)

            if frame is not None:
                stats.frames_ok += 1
                stats.last_err = ""
                self._set_stats(stats)
                return frame
            return None

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
