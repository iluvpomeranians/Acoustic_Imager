#!/usr/bin/env python3
"""
Monitor GPIO pins used by the acoustic imager. Prints a table in a loop.
Uses pinctrl so pin state is visible even when lines are in use by the kernel (SPI, I2C).
Run from repo root: python3 utilities/calibration/pin_monitor.py
  python3 utilities/calibration/pin_monitor.py --runs 5  # for calibration suite: exit 0 if all non-GND pins read 0/1
"""
import argparse
import os
import subprocess
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_SRC_SOFTWARE = os.path.join(_REPO_ROOT, "src", "software")
if _SRC_SOFTWARE not in sys.path:
    sys.path.insert(0, _SRC_SOFTWARE)

try:
    from acoustic_imager import config
except ImportError:
    config = None

# (physical_pin, bcm_gpio_or_None, usage). None = GND / no GPIO to read.
PINS_USED = [
    (22, 25, "Gain Ctrl"),
    (26, 7, "MCU status"),
    (27, 0, "RPi status"),
    (25, None, "GND"),
    (40, 21, "SCK"),
    (36, 16, "NSS (SPI1 CE2)"),
    (35, 19, "MISO"),
    (38, 20, "MOSI"),
    (39, None, "GND"),
    (28, 1, "NRST"),
]

def get_spi_path():
    """SPI device path from config or project default."""
    if config is not None and hasattr(config, "SPI_BUS") and hasattr(config, "SPI_DEV"):
        return f"/dev/spidev{config.SPI_BUS}.{config.SPI_DEV}"
    return "/dev/spidev1.2"


def spi_status(spi_path):
    """Return (exists: bool, in_use: bool). in_use=True if open() fails (e.g. EBUSY/EACCES)."""
    exists = os.path.exists(spi_path)
    in_use = False
    if exists:
        try:
            with open(spi_path, "rb") as f:
                pass
        except OSError:
            in_use = True
    return exists, in_use


def read_one(bcm):
    """Read pin state via pinctrl (no line request). Returns "0", "1", or "?"."""
    out = subprocess.run(
        ["pinctrl", "get", str(bcm)],
        capture_output=True,
        text=True,
        timeout=1,
    )
    if out.returncode != 0 or not out.stdout:
        return "?"
    for line in out.stdout.strip().splitlines():
        if "|" in line:
            part = line.split("|", 1)[1].strip().split()
            if part and part[0].lower() == "hi":
                return "1"
            if part and part[0].lower() == "lo":
                return "0"
    return "?"


def run_once():
    """One pass over all pins. Returns dict bcm -> value ('0', '1', '?') and SPI status."""
    results = {}
    for _physical, bcm, _usage in PINS_USED:
        if bcm is None:
            continue
        results[bcm] = read_one(bcm)
    return results


def main():
    ap = argparse.ArgumentParser(description="Monitor GPIO pins (pinctrl)")
    ap.add_argument("--runs", type=int, default=0, help="Run N times and exit; 0 = loop forever. For suite: 5")
    args = ap.parse_args()

    width = 58
    header = f"{'Physical':<10} {'BCM':<6} {'Usage':<26} {'Value':<6}"
    print(header)
    print("-" * width)

    runs_left = args.runs if args.runs > 0 else None
    bcms_with_valid_reading = set()  # non-GND bcms that got "0" or "1" at least once

    try:
        while runs_left is None or runs_left > 0:
            for physical, bcm, usage in PINS_USED:
                bcm_str = str(bcm) if bcm is not None else "-"
                if bcm is None:
                    val = "-"
                else:
                    val = read_one(bcm)
                    if val in ("0", "1"):
                        bcms_with_valid_reading.add(bcm)
                print(f"{physical:<10} {bcm_str:<6} {usage:<26} {val:<6}")
            spi_path = get_spi_path()
            exists, in_use = spi_status(spi_path)
            print(f"SPI: {spi_path}  exists={'yes' if exists else 'no'}  in_use={'yes' if in_use else 'no'}")
            print()

            if runs_left is not None:
                runs_left -= 1
                if runs_left <= 0:
                    break
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass

    if args.runs > 0:
        non_gnd_bcms = {bcm for _p, bcm, _u in PINS_USED if bcm is not None}
        all_ok = non_gnd_bcms <= bcms_with_valid_reading
        sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
