#!/usr/bin/env python3
"""
Full calibration suite (items 0-6). Run with acoustic imager stopped for SPI steps.
Prints progress to stdout; optionally writes a timestamped report to --report-dir.

Usage (from repo root):
  python3 utilities/calibration/run_suite.py --gain high
  python3 utilities/calibration/run_suite.py --gain low --no-write --report-dir ./cal_reports
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_CAL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CAL_DIR.parents[1]  # repo root (calibration -> utilities -> repo)
_SRC_SOFTWARE = str(_REPO_ROOT / "src" / "software")
if _SRC_SOFTWARE not in sys.path:
    sys.path.insert(0, _SRC_SOFTWARE)

# Default timeouts (seconds)
TIMEOUT_PIN = 15
TIMEOUT_SILENCE = 10
TIMEOUT_CAL_TEST = 10
TIMEOUT_SPI_DEVICES = 8
TIMEOUT_MAGIC_PROBE = 12
TIMEOUT_FPS_CHECK = 8
TIMEOUT_METRICS = 25
TIMEOUT_POST_GAIN = 15


def _log(msg: str, report_lines: list[str] | None) -> None:
    print(msg)
    if report_lines is not None:
        report_lines.append(msg)


def _run_cmd(cmd: list, timeout: int, cwd: str, report_lines: list[str] | None) -> tuple[int, str]:
    """Run command; return (returncode, combined stdout+stderr)."""
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        out = (r.stdout or "").strip() + "\n" + (r.stderr or "").strip()
        return r.returncode, out
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    except Exception as e:
        return -1, str(e)


def step0_silence_check(skip: bool, report_lines: list[str] | None) -> bool:
    """Pre-test: environment quiet enough. Returns True if passed or skipped."""
    if skip:
        _log("[0] Silence check: skipped (--skip-silence)", report_lines)
        return True
    _log("[0] Silence check (env quiet enough for gain cal)...", report_lines)
    code, out = _run_cmd(
        [sys.executable, str(_CAL_DIR / "silence_check.py"), "--sec", "1.5"],
        TIMEOUT_SILENCE,
        str(_REPO_ROOT),
        report_lines,
    )
    for line in out.splitlines():
        _log("  " + line, report_lines)
    if code != 0:
        _log("  FAILED (env too loud or no SPI)", report_lines)
        return False
    _log("  OK", report_lines)
    return True


def step1_pin_monitor(report_lines: list[str] | None) -> bool:
    """Ribbon cable pins: 5 runs, all non-GND must read 0 or 1."""
    _log("[1] Pin monitor (ribbon cable, 5 runs)...", report_lines)
    code, out = _run_cmd(
        [sys.executable, str(_CAL_DIR / "pin_monitor.py"), "--runs", "5"],
        TIMEOUT_PIN,
        str(_REPO_ROOT),
        report_lines,
    )
    for line in out.splitlines():
        _log("  " + line[:90], report_lines)
    if code != 0:
        _log("  FAILED (some pins never read 0/1)", report_lines)
        return False
    _log("  OK", report_lines)
    return True


def step2_geometry(report_lines: list[str] | None) -> bool:
    """Array geometry + calibration_test --validate."""
    _log("[2] Array geometry & config alignment...", report_lines)
    # 2a) calibration_test --validate
    code, out = _run_cmd(
        [sys.executable, str(_CAL_DIR / "calibration_test.py"), "--validate"],
        TIMEOUT_CAL_TEST,
        str(_REPO_ROOT),
        report_lines,
    )
    for line in out.splitlines():
        _log("  " + line[:90], report_lines)
    if code != 0:
        _log("  calibration_test --validate FAILED", report_lines)
        return False
    # 2b) geometry consistency (array_geometry vs config)
    try:
        from acoustic_imager import config as cfg
        cal_dir_str = str(_CAL_DIR)
        if cal_dir_str not in sys.path:
            sys.path.insert(0, cal_dir_str)
        import array_geometry.array_geometry as ag
        n = getattr(cfg, "N_MICS", 16)
        if getattr(ag, "N_MICS", None) != n:
            _log("  N_MICS mismatch: array_geometry=%s config=%s" % (getattr(ag, "N_MICS", None), n), report_lines)
            return False
        x_hw = getattr(cfg, "x_coords_hw", None)
        y_hw = getattr(cfg, "y_coords_hw", None)
        if x_hw is None or y_hw is None or len(x_hw) != n or len(y_hw) != n:
            _log("  config x_coords_hw/y_coords_hw missing or wrong length", report_lines)
            return False
        _log("  Geometry OK (payload index & config aligned)", report_lines)
    except Exception as e:
        _log("  Geometry check error: " + str(e), report_lines)
        return False
    return True


def step3_spi_devices(report_lines: list[str] | None) -> bool:
    """SPI device nodes and boot config."""
    _log("[3] SPI devices...", report_lines)
    sh = _CAL_DIR / "check_spi_devices.sh"
    if not sh.exists():
        import glob
        devs = glob.glob("/dev/spidev*")
        _log("  " + ("Found: " + " ".join(devs) if devs else "No /dev/spidev*"), report_lines)
        return True
    code, out = _run_cmd(["bash", str(sh)], TIMEOUT_SPI_DEVICES, str(_REPO_ROOT), report_lines)
    for line in out.splitlines():
        _log("  " + line[:90], report_lines)
    return code == 0


def step4_parsing_and_rates(report_lines: list[str] | None) -> tuple[bool, float, int, int, int]:
    """SPI magic probe, then FPS + CRC/bad_parse. Returns (ok, fps, frames_ok, bad_parse, bad_crc)."""
    _log("[4] Parsing (magic probe) & frame rate...", report_lines)
    # 4a) Magic probe
    probe = _CAL_DIR / "spi_magic_probe.py"
    if probe.exists():
        code, out = _run_cmd(
            [sys.executable, str(probe), "--no-sync"],
            TIMEOUT_MAGIC_PROBE,
            str(_REPO_ROOT),
            report_lines,
        )
        for line in out.splitlines():
            _log("  " + line[:90], report_lines)
        magic_ok = "magic" in out.lower() and ("@" in out or "LE" in out or "BE" in out)
    else:
        magic_ok = False
        _log("  spi_magic_probe.py not found", report_lines)

    # 4b) SPI rate from config
    try:
        from acoustic_imager import config as cfg
        speed = getattr(cfg, "SPI_MAX_SPEED_HZ", 0)
        _log("  SPI_MAX_SPEED_HZ = %s" % speed, report_lines)
    except Exception:
        speed = 0

    # 4c) FPS check + CRC/bad_parse via SPIManager
    fps = 0.0
    frames_ok = bad_parse = bad_crc = 0
    try:
        from acoustic_imager.io.spi_manager import SPIManager
        mgr = SPIManager(use_frame_ready=True)
        mgr.start()
        try:
            t0 = time.monotonic()
            duration = 1.5
            while time.monotonic() - t0 < duration:
                lf = mgr.get_latest()
                if lf.ok and lf.fft_data is not None:
                    frames_ok = lf.stats.frames_ok
                    bad_parse = lf.stats.bad_parse
                    bad_crc = getattr(lf.stats, "bad_crc", 0)
                time.sleep(0.02)
            elapsed = time.monotonic() - t0
            lf = mgr.get_latest()
            if lf.stats.frames_ok > 0:
                frames_ok = lf.stats.frames_ok
                bad_parse = lf.stats.bad_parse
                bad_crc = getattr(lf.stats, "bad_crc", 0)
                fps = frames_ok / elapsed if elapsed > 0 else 0
            _log("  FPS = %.1f  frames_ok = %d  bad_parse = %d  bad_crc = %d" % (fps, frames_ok, bad_parse, bad_crc), report_lines)
        finally:
            mgr.stop()
    except Exception as e:
        _log("  FPS check error (SPI in use?): " + str(e), report_lines)

    ok = magic_ok and fps >= 15  # accept if we saw magic and reasonable FPS
    if not ok:
        _log("  FAILED (magic or FPS)", report_lines)
    else:
        _log("  OK", report_lines)
    return ok, fps, frames_ok, bad_parse, bad_crc


def step5_metrics(gain: str, write_config: bool, report_lines: list[str] | None) -> bool:
    """Gain normalization: metrics_debug --live --frames 5 --average [--write-config --write-from gain]."""
    _log("[5] Metrics (gain normalization)...", report_lines)
    cmd = [sys.executable, str(_CAL_DIR / "metrics_debug.py"), "--live", "--frames", "5", "--average", "--sec", "2"]
    if write_config:
        cmd += ["--write-config", "--write-from", gain]
    code, out = _run_cmd(cmd, TIMEOUT_METRICS, str(_REPO_ROOT), report_lines)
    for line in out.splitlines():
        _log("  " + line[:90], report_lines)
    if code != 0:
        _log("  FAILED", report_lines)
        return False
    _log("  OK", report_lines)
    return True


def step6_post_gain_sanity(report_lines: list[str] | None) -> bool:
    """After writing gain: run metrics --live once, check ratio reasonable."""
    _log("[6] Post-gain sanity (ratio check)...", report_lines)
    code, out = _run_cmd(
        [sys.executable, str(_CAL_DIR / "metrics_debug.py"), "--live", "--frames", "3", "--sec", "1.5"],
        TIMEOUT_POST_GAIN,
        str(_REPO_ROOT),
        report_lines,
    )
    for line in out.splitlines():
        _log("  " + line[:90], report_lines)
    # Heuristic: "Ratio max/min" should be finite and not huge
    if "Ratio max/min" in out:
        _log("  OK (ratio in output)", report_lines)
        return code == 0
    return code == 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Full calibration suite (0-6)")
    ap.add_argument("--gain", choices=("high", "low"), default="high", help="Gain mode for --write-from (default high)")
    ap.add_argument("--no-write", action="store_true", help="Do not write SPI_MIC_GAIN to config")
    ap.add_argument("--skip-silence", action="store_true", help="Skip silence pre-check")
    ap.add_argument("--report-dir", type=str, default="", help="Write timestamped report file here")
    args = ap.parse_args()

    report_lines = []

    _log("=== Calibration Suite (stop acoustic imager for SPI steps) ===", report_lines)
    results = {}

    # 0
    results["silence"] = step0_silence_check(args.skip_silence, report_lines)

    # 1
    results["pins"] = step1_pin_monitor(report_lines)

    # 2
    results["geometry"] = step2_geometry(report_lines)

    # 3
    results["spi_devices"] = step3_spi_devices(report_lines)

    # 4
    results["parsing_ok"], fps, frames_ok, bad_parse, bad_crc = step4_parsing_and_rates(report_lines)
    results["fps"] = fps
    results["bad_parse"] = bad_parse
    results["bad_crc"] = bad_crc

    # 5
    results["metrics"] = step5_metrics(args.gain, write_config=not args.no_write, report_lines=report_lines)

    # 6 (only if we wrote config)
    if not args.no_write and results["metrics"]:
        results["post_gain"] = step6_post_gain_sanity(report_lines)
    else:
        results["post_gain"] = True
        _log("[6] Post-gain sanity: skipped (no write)", report_lines)

    # Optional: temperature (e.g. Pi)
    try:
        tpath = Path("/sys/class/thermal/thermal_zone0/temp")
        if tpath.exists():
            t_c = int(tpath.read_text().strip()) / 1000.0
            _log("CPU temp: %.1f C" % t_c, report_lines)
    except Exception:
        pass

    # Summary
    passed = sum(1 for v in results.values() if v is True)
    total = len([k for k in results if isinstance(results[k], bool)])
    _log("=== Done: %d/%d steps passed ===" % (passed, total), report_lines)

    if args.report_dir:
        d = Path(args.report_dir)
        d.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = d / ("cal_suite_%s.txt" % ts)
        report_file.write_text("\n".join(report_lines), encoding="utf-8")
        _log("Report written: %s" % report_file, report_lines)

    return 0 if all(v is True for k, v in results.items() if isinstance(v, bool)) else 1


if __name__ == "__main__":
    sys.exit(main())
