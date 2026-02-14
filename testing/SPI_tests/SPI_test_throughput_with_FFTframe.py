import spidev
import time
import zlib
import sys
import struct
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "dataframe"))
from fftframe import FFTFrame  # optional; not required for throughput test

# ---------------------------
# Config
# ---------------------------
N_MICS = 16
SAMPLE_RATE_HZ = 100000
SAMPLES_PER_CHANNEL = 1024

# rFFT bins for real FFT: N/2 + 1
N_BINS = SAMPLES_PER_CHANNEL // 2 + 1  # 513

FPS_TARGET = 30
CHUNK = 4096
SPI_HZ = 16_000_000
DURATION_S = 100

# ---------------------------
# Frame format (example)
# ---------------------------
MAGIC_START = 0x46524654  # 'TF RF' (just a tag) - pick anything stable
MAGIC_END   = 0x454E4421  # 'END!'

VERSION = 1

# Little-endian header:
# magic(u32), ver(u16), hdr_len(u16),
# seq(u32),
# mic_count(u16), fft_size(u16),
# sample_rate(u32),
# bins(u16), reserved(u16),
# payload_len(u32)
HEADER_FMT = "<IHH I HH I HH I"
HEADER_LEN = struct.calcsize(HEADER_FMT)

# Trailer: crc32(u32), end_magic(u32)
TRAILER_FMT = "<II"
TRAILER_LEN = struct.calcsize(TRAILER_FMT)

PAYLOAD_LEN = N_MICS * N_BINS * 4  # float32
FRAME_BYTES = HEADER_LEN + PAYLOAD_LEN + TRAILER_LEN


def make_payload(seq: int) -> bytes:
    """
    Deterministic payload shaped like (N_MICS, N_BINS) float32.
    This emulates '513 bins * 4 bytes * 16 channels'.
    """
    # Create stable-but-changing values based on seq
    # (fast to generate; no trig)
    base = (seq % 1024) * 0.01
    arr = np.empty((N_MICS, N_BINS), dtype=np.float32)
    for ch in range(N_MICS):
        arr[ch, :] = base + ch + (np.arange(N_BINS, dtype=np.float32) * 1e-4)
    return arr.tobytes(order="C")


def make_frame(seq: int) -> bytearray:
    payload = make_payload(seq)

    header = struct.pack(
        HEADER_FMT,
        MAGIC_START,
        VERSION,
        HEADER_LEN,
        seq,
        N_MICS,
        SAMPLES_PER_CHANNEL,
        SAMPLE_RATE_HZ,
        N_BINS,
        0,  # reserved
        len(payload),
    )

    # CRC over header + payload (common embedded pattern)
    crc = zlib.crc32(header)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF

    trailer = struct.pack(TRAILER_FMT, crc, MAGIC_END)

    return bytearray(header + payload + trailer)


def xfer_bytes(spi: spidev.SpiDev, tx: memoryview) -> bytearray:
    rx = bytearray(len(tx))
    offset = 0
    while offset < len(tx):
        end = min(offset + CHUNK, len(tx))
        r = spi.xfer2(list(tx[offset:end]))
        rx[offset:end] = bytes(r)
        offset = end
    return rx


def quick_validate_frame(buf: bytes) -> bool:
    """Basic check: header magic, lengths, end magic, crc."""
    if len(buf) != FRAME_BYTES:
        return False

    # Parse header
    try:
        (magic, ver, hdr_len, seq, mic, fft_size, fs, bins, _res, pay_len) = struct.unpack_from(
            HEADER_FMT, buf, 0
        )
    except struct.error:
        return False

    if magic != MAGIC_START or ver != VERSION or hdr_len != HEADER_LEN:
        return False
    if mic != N_MICS or fft_size != SAMPLES_PER_CHANNEL or fs != SAMPLE_RATE_HZ or bins != N_BINS:
        return False
    if pay_len != PAYLOAD_LEN:
        return False

    payload_off = HEADER_LEN
    trailer_off = HEADER_LEN + PAYLOAD_LEN

    (crc_rx, magic_end) = struct.unpack_from(TRAILER_FMT, buf, trailer_off)
    if magic_end != MAGIC_END:
        return False

    header = buf[:HEADER_LEN]
    payload = buf[payload_off:payload_off + PAYLOAD_LEN]
    crc = zlib.crc32(header)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF

    return crc == crc_rx


def main():
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.mode = 0
    spi.bits_per_word = 8
    spi.max_speed_hz = SPI_HZ

    print(f"SPI Hz: {SPI_HZ:,}")
    print(f"Frame bytes (with hdr+crc+footer): {FRAME_BYTES}")
    print(f"Target FPS: {FPS_TARGET}  | Target rate: {FRAME_BYTES*FPS_TARGET/1e6:.3f} MB/s")
    print(f"Chunk size: {CHUNK}")
    print("Running... (no printing in the loop)")

    t_start = time.perf_counter()
    t_end = t_start + DURATION_S

    frames = 0
    bad_loopback = 0
    bad_parse = 0
    bytes_total = 0
    next_deadline = t_start

    try:
        seq = 0
        while True:
            now = time.perf_counter()
            if now >= t_end:
                break

            next_deadline += 1.0 / FPS_TARGET
            if now < next_deadline:
                time.sleep(max(0.0, next_deadline - now - 0.0003))

            tx = make_frame(seq)
            rx = xfer_bytes(spi, memoryview(tx))

            if rx != tx:
                bad_loopback += 1

            # Optional: validate the received frame structure/CRC
            if not quick_validate_frame(rx):
                bad_parse += 1

            frames += 1
            bytes_total += FRAME_BYTES
            seq += 1

    finally:
        spi.close()

    elapsed = time.perf_counter() - t_start
    mb_s = bytes_total / elapsed / 1e6
    mbps = (bytes_total * 8) / elapsed / 1e6
    fps = frames / elapsed

    print("\nRESULTS")
    print(f"Elapsed: {elapsed:.3f} s")
    print(f"Frames:  {frames}  | FPS achieved: {fps:.2f}")
    print(f"Data:    {bytes_total/1e6:.3f} MB  | Throughput: {mb_s:.3f} MB/s ({mbps:.3f} Mb/s)")
    print(f"Loopback mismatches: {bad_loopback}")
    print(f"Frame validate fails: {bad_parse}")


if __name__ == "__main__":
    main()
